use clap::{Arg, Command};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[inline(always)]
fn increment_nonce(buffer: &mut [u8; 64], start: usize) -> usize {
    let mut i = start;
    while i > 0 {
        if buffer[i - 1] == b'9' {
            buffer[i - 1] = b'0';
            i -= 1;
        } else {
            buffer[i - 1] += 1;
            break;
        }
    }
    if i == 0 {
        buffer[start] = b'1';
        start + 1
    } else {
        start
    }
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

    let prefix_bytes = prefix.as_bytes();

    let global_best_score = Arc::new(AtomicUsize::new(0));
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
                (Sha256::new(), [0u8; 64]),
                |(hasher, buffer), chunk_start| {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            prefix_bytes.as_ptr(),
                            buffer.as_mut_ptr(),
                            prefix_bytes.len(),
                        );
                    }
                    let mut nonce_start = prefix_bytes.len();
                    nonce_start +=
                        get_nonce_start_delta(chunk_start as u64, &mut buffer[nonce_start..]);

                    let chunk_end = (chunk_start + 16384).min(range_end);
                    let mut local_best_score = 0;
                    let mut local_best_nonce = 0;
                    let mut local_best_hash = [0u8; 32];

                    for n in chunk_start..chunk_end {
                        hasher.update(&buffer[..nonce_start]);
                        let result = hasher.finalize_reset();

                        let score = result.iter().take(4).fold(0, |acc, &byte| {
                            if byte == 0 {
                                acc + 2
                            } else if byte < 16 {
                                acc + 1
                            } else {
                                acc
                            }
                        });

                        if score > local_best_score
                            || (score == local_best_score && n < local_best_nonce)
                        {
                            local_best_score = score;
                            local_best_nonce = n;
                            local_best_hash.copy_from_slice(&result);
                        }

                        nonce_start = increment_nonce(buffer, nonce_start);
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
                                "\nNew min with {} zeroes: {} -> {}\n",
                                local_best_score,
                                format!("{}{}", prefix, local_best_nonce),
                                hex::encode(&*global_hash)
                            );
                        }
                    }
                },
            );

        let elapsed = begin.elapsed();
        let hps = batch_size as f64 / elapsed.as_secs_f64();

        current_start.fetch_add(batch_size as usize, Ordering::Relaxed);

        println!(
            "{} @ {:5.2} MH/s :: ({}) {} -> {}",
            range_end,
            hps / 1e6,
            global_best_score.load(Ordering::Relaxed),
            format!("{}{}", prefix, global_best_nonce.load(Ordering::Relaxed)),
            hex::encode(&*global_best_hash.read())
        );
    }
}
