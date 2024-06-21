use clap::{Arg, Command};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[inline(always)]
fn nonce_to_bytes(n: u64, buffer: &mut [u8]) -> usize {
    if n < 10 {
        buffer[0] = n as u8 + b'0';
        return 1;
    }
    let mut len = 0;
    let mut num = n;
    while num > 0 {
        buffer[len] = (num % 10) as u8 + b'0';
        num /= 10;
        len += 1;
    }
    buffer[..len].reverse();
    len
}

#[inline(always)]
fn score(hash: &[u8; 32]) -> u32 {
    let mut score = 0;
    for &byte in hash.iter().take(4) {
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

    let prefix_bytes = prefix.as_bytes();

    let global_best_score = Arc::new(AtomicUsize::new(0));
    let global_best_nonce = Arc::new(AtomicUsize::new(0));
    let global_best_hash = Arc::new(parking_lot::RwLock::new(String::new()));
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
                    buffer[..prefix_bytes.len()].copy_from_slice(prefix_bytes);
                    let chunk_end = (chunk_start + 16384).min(range_end);
                    let mut local_best_score = 0;
                    let mut local_best_nonce = 0;
                    let mut local_best_hash = [0u8; 32];

                    for n in chunk_start..chunk_end {
                        let prefix_len = prefix_bytes.len();
                        let nonce_len = nonce_to_bytes(n as u64, &mut buffer[prefix_len..]);
                        hasher.update(&buffer[..prefix_len + nonce_len]);
                        let result: [u8; 32] = hasher.finalize_reset().into();
                        let s = score(&result);

                        if s > local_best_score || (s == local_best_score && n < local_best_nonce) {
                            local_best_score = s;
                            local_best_nonce = n;
                            local_best_hash = result;
                        }
                    }

                    let current_best_score = global_best_score.load(Ordering::Relaxed);
                    if local_best_score > current_best_score as u32
                        || (local_best_score == current_best_score as u32
                            && local_best_nonce < global_best_nonce.load(Ordering::Relaxed))
                    {
                        if global_best_score
                            .compare_exchange_weak(
                                current_best_score,
                                local_best_score as usize,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            global_best_nonce.store(local_best_nonce, Ordering::Relaxed);
                            let hash_str = hex::encode(&local_best_hash);
                            let mut global_hash = global_best_hash.write();
                            *global_hash = hash_str;

                            println!(
                                "\nNew min with {} zeroes: {} -> {}\n",
                                local_best_score,
                                format!("{}{}", prefix, local_best_nonce),
                                *global_hash
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
            global_best_hash.read().clone()
        );
    }
}
