use clap::{Arg, Command};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

const MAX_PREFIX_LEN: usize = 64;
const MAX_NONCE_LEN: usize = 64;
const BUFFER_LEN: usize = MAX_PREFIX_LEN + MAX_NONCE_LEN;

#[inline(always)]
fn increment_nonce(buffer: &mut [u8], start: usize, end: usize) -> usize {
    let mut i = end - 1;
    while i >= start {
        if buffer[i] == b'9' {
            buffer[i] = b'0';
            if i == start {
                for j in (start + 1..=end).rev() {
                    buffer[j] = buffer[j - 1];
                }
                buffer[start] = b'1';
                return end + 1;
            }
        } else {
            buffer[i] += 1;
            return end;
        }
        i -= 1;
    }
    end
}

#[inline(always)]
fn get_nonce_start_delta(mut n: u64, buf: &mut [u8]) -> usize {
    if n == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut i = 0;
    while n > 0 {
        buf[i] = (n % 10) as u8 + b'0';
        n /= 10;
        i += 1;
    }
    buf[..i].reverse();
    i
}

#[inline(always)]
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize] as char);
        s.push(HEX_CHARS[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn calculate_score(result: &[u8; 32]) -> u32 {
    let mut score = 0u32;
    let zero = _mm_setzero_si128();
    let low_mask = _mm_set1_epi8(0x0F);

    for chunk in result.chunks(16) {
        let v = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
        let full_zero_mask = _mm_cmpeq_epi8(v, zero);
        let full_zero_bits = _mm_movemask_epi8(full_zero_mask) as u32;

        if full_zero_bits != 0xFFFF {
            let low_nibbles = _mm_and_si128(v, low_mask);
            let half_zero_mask = _mm_cmpeq_epi8(low_nibbles, zero);
            let half_zero_bits = _mm_movemask_epi8(half_zero_mask) as u32;

            let leading_full_zeros = full_zero_bits.trailing_ones();
            score += leading_full_zeros * 2;

            if (half_zero_bits & !full_zero_bits) & (1 << leading_full_zeros) != 0 {
                score += 1;
            }
            break;
        }
        score += 32;
    }
    score
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn calculate_score(result: &[u8; 32]) -> u32 {
    let mut score = 0u32;

    for &byte in result.iter() {
        if byte == 0 {
            score += 2;
        } else {
            if byte < 16 {
                score += 1;
            }
            break;
        }
    }

    score
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    let matches = Command::new("shallenge")
        .version("1.0")
        .about("Bruteforce the lowest SHA256 hash")
        .arg(
            Arg::new("start")
                .short('s')
                .long("start")
                .value_name("START")
                .help("Starting value for the nonce")
                .default_value("0"),
        )
        .arg(
            Arg::new("batch-size")
                .short('b')
                .long("batch-size")
                .value_name("BATCH_SIZE")
                .help("Size of the batch to process")
                .default_value("500000000"),
        )
        .arg(
            Arg::new("prefix")
                .short('p')
                .long("prefix")
                .value_name("PREFIX")
                .help("Prefix to use for the hash input")
                .required(true),
        )
        .get_matches();

    let start: u64 = matches
        .get_one::<String>("start")
        .unwrap()
        .parse()
        .expect("Invalid start value");

    let batch_size: u64 = matches
        .get_one::<String>("batch-size")
        .unwrap()
        .parse()
        .expect("Invalid batch size");

    let prefix = matches
        .get_one::<String>("prefix")
        .expect("prefix argument is required");

    let prefix_len = prefix.len();
    if prefix_len > MAX_PREFIX_LEN {
        panic!("Prefix is too long. Maximum length is {}", MAX_PREFIX_LEN);
    }

    let global_best_score = Arc::new(AtomicU32::new(0));
    let global_best_nonce = Arc::new(AtomicUsize::new(0));
    let global_best_hash = Arc::new(parking_lot::RwLock::new([0u8; 32]));
    let current_start = Arc::new(AtomicUsize::new(start as usize));

    loop {
        let range_start = current_start.load(Ordering::Relaxed);
        let range_end = range_start + batch_size as usize;

        let begin = Instant::now();

        (range_start..range_end)
            .into_par_iter()
            .step_by(16384)
            .for_each_with(
                (Sha256::new(), [0u8; BUFFER_LEN]),
                |(hasher, buffer), chunk_start| {
                    buffer[..prefix_len].copy_from_slice(prefix.as_bytes());
                    let nonce_start = prefix_len;
                    let mut nonce_end = nonce_start
                        + get_nonce_start_delta(chunk_start as u64, &mut buffer[nonce_start..]);

                    let chunk_end = (chunk_start + 16384).min(range_end);
                    let mut local_best_score = 0;
                    let mut local_best_nonce = 0;
                    let mut local_best_hash = [0u8; 32];

                    for n in chunk_start..chunk_end {
                        hasher.update(&buffer[..nonce_end]);
                        let result = hasher.finalize_reset();

                        let score = unsafe { calculate_score(result.as_ref()) };

                        if score > local_best_score
                            || (score == local_best_score && n < local_best_nonce)
                        {
                            local_best_score = score;
                            local_best_nonce = n;
                            local_best_hash.copy_from_slice(&result);
                        }

                        nonce_end = increment_nonce(buffer, nonce_start, nonce_end);
                    }

                    let current_best_score = global_best_score.load(Ordering::Relaxed);
                    if local_best_score > current_best_score
                        || (local_best_score == current_best_score
                            && local_best_nonce < global_best_nonce.load(Ordering::Relaxed))
                    {
                        if global_best_score
                            .compare_exchange(
                                current_best_score,
                                local_best_score,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            global_best_nonce.store(local_best_nonce, Ordering::Relaxed);
                            let mut global_hash = global_best_hash.write();
                            *global_hash = local_best_hash;

                            println!(
                                "\nNew min with {} zeroes: {}{} -> {}\n",
                                local_best_score,
                                prefix,
                                local_best_nonce,
                                hex_encode(&local_best_hash)
                            );
                        }
                    }
                },
            );

        let elapsed = begin.elapsed();
        let hps = batch_size as f64 / elapsed.as_secs_f64();

        current_start.fetch_add(batch_size as usize, Ordering::Relaxed);

        println!(
            "{} @ {:5.2} MH/s :: ({}) {}{} -> {}",
            range_end,
            hps / 1e6,
            global_best_score.load(Ordering::Relaxed),
            prefix,
            global_best_nonce.load(Ordering::Relaxed),
            hex_encode(&*global_best_hash.read())
        );
    }
}
