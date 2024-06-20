use clap::{Arg, Command};
use rayon::prelude::*;
use ring::digest::{digest, SHA256};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn hash(prefix: &[u8], n: usize, buffer: &mut [u8; 64]) -> [u8; 32] {
    let prefix_len = prefix.len();
    buffer[..prefix_len].copy_from_slice(prefix);

    let mut num = n;
    let mut pos = prefix_len;
    let mut digits = [0u8; 20];
    let mut i = digits.len();

    loop {
        i -= 1;
        digits[i] = b'0' + (num % 10) as u8;
        num /= 10;
        if num == 0 {
            break;
        }
    }

    let digit_count = digits.len() - i;
    buffer[pos..pos + digit_count].copy_from_slice(&digits[i..]);
    pos += digit_count;

    let digest = digest(&SHA256, &buffer[..pos]);
    let mut hash_result = [0u8; 32];
    hash_result.copy_from_slice(digest.as_ref());
    hash_result
}

fn score(hash: &[u8; 32]) -> usize {
    let mut score = 0;
    for &byte in hash.iter() {
        if byte == 0 {
            score += 2;
        } else {
            if byte >> 4 == 0 {
                score += 1;
            }
            break;
        }
    }
    score
}

fn main() {
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

    let start = matches
        .get_one::<String>("start")
        .unwrap()
        .parse::<usize>()
        .expect("Invalid start value");

    let batch_size = matches
        .get_one::<String>("batch-size")
        .unwrap()
        .parse::<usize>()
        .expect("Invalid batch size");

    let prefix = matches
        .get_one::<String>("prefix")
        .expect("prefix argument is required")
        .as_bytes();

    let mut global_best_score = 0;
    let global_best_hash = Arc::new(Mutex::new(String::new()));
    let mut current_start = start;

    loop {
        let range = current_start..current_start + batch_size;

        let begin = Instant::now();
        let (local_best_nonce, local_best_hash, local_best_score) = range
            .into_par_iter()
            .map_with([0u8; 64], |buffer, n| {
                let h = hash(prefix, n, buffer);
                let s = score(&h);
                (n, h, s)
            })
            .reduce_with(|a, b| if a.2 > b.2 { a } else { b })
            .unwrap();

        let best_hash_str = hex::encode(&local_best_hash);
        let elapsed = begin.elapsed();
        let hps = batch_size as f64 / elapsed.as_secs_f64();

        current_start += batch_size;

        println!("{} @ {:5.2} MH/s", current_start, hps / 1e6);

        let mut global_best_hash_guard = global_best_hash.lock().unwrap();
        if (local_best_score > global_best_score)
            || (local_best_score == global_best_score && best_hash_str < *global_best_hash_guard)
        {
            println!(
                "New min with {} zeroes: {} -> {}",
                local_best_score,
                format!("{}{}", String::from_utf8_lossy(prefix), local_best_nonce),
                best_hash_str
            );
            global_best_score = local_best_score;
            *global_best_hash_guard = best_hash_str.clone();
        }
    }
}
