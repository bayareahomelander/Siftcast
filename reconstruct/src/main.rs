use siftcast_reconstruct::{reconstruct_bytes, write_jsonl};
use std::env;
use std::fs;
use std::process;

fn main() {
    let mut input = String::from("artifacts/capture.bin");
    let mut output = String::from("artifacts/series.jsonl");
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--in" => {
                input = args.next().expect("--in value");
            }
            "--out" => {
                output = args.next().expect("--out value");
            }
            other => {
                eprintln!("unknown arg {other}");
                process::exit(2);
            }
        }
    }
    let bytes = fs::read(&input).unwrap_or_else(|e| {
        eprintln!("reconstruct: read {input}: {e}");
        process::exit(1);
    });
    let rec = reconstruct_bytes(&bytes);
    if let Err(e) = write_jsonl(&output, &rec.out) {
        eprintln!("reconstruct: write {output}: {e}");
        process::exit(1);
    }
    let trusted = rec.out.iter().filter(|r| r.trust >= 0.5).count();
    let held = rec.out.iter().filter(|r| r.held).count();
    eprintln!(
        "reconstruct: bytes={} rows={} trusted={} held={} crc_fail={} dups_old={} sync_hits={}",
        rec.stats.bytes,
        rec.out.len(),
        trusted,
        held,
        rec.stats.crc_fail,
        rec.stats.dups_old,
        rec.stats.sync_hits
    );
}
