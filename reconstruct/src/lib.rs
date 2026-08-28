//! Streaming CRC-validated reorder reconstruct of the 13-byte siftcast frame.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Write};

pub const SYNC0: u8 = 0xA5;
pub const SYNC1: u8 = 0x5A;
pub const FRAME_BODY: usize = 11; // seq..crc after the two sync bytes
pub const REORDER_WINDOW: u32 = 16;

// CRC-8/SMBUS, poly 0x07, init 0. Must match sim.cpp.
pub fn crc8(data: &[u8]) -> u8 {
    let mut c: u8 = 0;
    for &b in data {
        c ^= b;
        for _ in 0..8 {
            c = if c & 0x80 != 0 {
                (c << 1) ^ 0x07
            } else {
                c << 1
            };
        }
    }
    c
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub seq: u32,
    pub t_ms: u32,
    pub x: f64,
    pub y: f64,
    pub trust: f64,
    pub crc_ok: bool,
    pub held: bool,
}

#[derive(Default)]
pub struct Stats {
    pub bytes: usize,
    pub sync_hits: usize,
    pub crc_fail: usize,
    pub dups_old: usize,
}

pub struct ByteParser {
    st: u8,
    buf: Vec<u8>,
}

impl Default for ByteParser {
    fn default() -> Self {
        Self {
            st: 0,
            buf: Vec::with_capacity(FRAME_BODY),
        }
    }
}

impl ByteParser {
    pub fn push(&mut self, b: u8) -> Option<[u8; FRAME_BODY]> {
        match self.st {
            0 => {
                if b == SYNC0 {
                    self.st = 1;
                }
            }
            1 => {
                if b == SYNC1 {
                    self.st = 2;
                    self.buf.clear();
                } else if b == SYNC0 {
                    self.st = 1;
                } else {
                    self.st = 0;
                }
            }
            _ => {
                self.buf.push(b);
                if self.buf.len() == FRAME_BODY {
                    self.st = 0;
                    let mut body = [0u8; FRAME_BODY];
                    body.copy_from_slice(&self.buf);
                    return Some(body);
                }
            }
        }
        None
    }
}

pub fn parse_body(body: &[u8; FRAME_BODY]) -> (u32, u32, f64, f64, bool) {
    let seq = u16::from_le_bytes([body[0], body[1]]) as u32;
    let t_ms = u32::from_le_bytes([body[2], body[3], body[4], body[5]]);
    let xi = i16::from_le_bytes([body[6], body[7]]);
    let yi = i16::from_le_bytes([body[8], body[9]]);
    let crc_ok = crc8(&body[..10]) == body[10];
    (seq, t_ms, xi as f64 / 100.0, yi as f64 / 100.0, crc_ok)
}

pub struct Reconstructor {
    expected: u32,
    last_x: f64,
    last_y: f64,
    last_t: u32,
    window: BTreeMap<u32, (u32, f64, f64)>,
    pub out: Vec<Record>,
    max_seen: i64,
    pub stats: Stats,
}

impl Default for Reconstructor {
    fn default() -> Self {
        Self {
            expected: 0,
            last_x: 0.0,
            last_y: 0.0,
            last_t: 0,
            window: BTreeMap::new(),
            out: Vec::new(),
            max_seen: -1,
            stats: Stats::default(),
        }
    }
}

impl Reconstructor {
    fn emit(&mut self, seq: u32, t_ms: u32, x: f64, y: f64, trust: f64, crc_ok: bool, held: bool) {
        self.last_x = x;
        self.last_y = y;
        self.last_t = t_ms;
        self.out.push(Record {
            seq,
            t_ms,
            x,
            y,
            trust,
            crc_ok,
            held,
        });
    }

    fn emit_hold(&mut self) {
        self.emit(
            self.expected,
            self.last_t,
            self.last_x,
            self.last_y,
            0.0,
            false,
            true,
        );
        self.expected += 1;
    }

    fn drain(&mut self) {
        while let Some((t, x, y)) = self.window.remove(&self.expected) {
            self.emit(self.expected, t, x, y, 1.0, true, false);
            self.expected += 1;
        }
    }

    fn stall_fill(&mut self) {
        loop {
            let too_far = self
                .window
                .keys()
                .next()
                .copied()
                .map(|s| s > self.expected + REORDER_WINDOW)
                .unwrap_or(false);
            if !too_far && self.window.len() <= REORDER_WINDOW as usize {
                break;
            }
            if self.window.contains_key(&self.expected) {
                self.drain();
                continue;
            }
            self.emit_hold();
            self.drain();
        }
    }

    pub fn push_frame(&mut self, seq: u32, t_ms: u32, x: f64, y: f64, crc_ok: bool) {
        self.stats.sync_hits += 1;
        if !crc_ok {
            self.stats.crc_fail += 1;
            return;
        }
        if seq < self.expected {
            self.stats.dups_old += 1;
            return;
        }
        self.max_seen = self.max_seen.max(seq as i64);
        if seq == self.expected {
            self.emit(seq, t_ms, x, y, 1.0, true, false);
            self.expected += 1;
            self.drain();
        } else {
            self.window.insert(seq, (t_ms, x, y));
            self.stall_fill();
        }
    }

    pub fn finish(&mut self) {
        let end = (self.max_seen + 1).max(self.expected as i64) as u32;
        while self.expected < end {
            if self.window.contains_key(&self.expected) {
                self.drain();
            } else {
                self.emit_hold();
                self.drain();
            }
        }
    }
}

pub fn reconstruct_bytes(bytes: &[u8]) -> Reconstructor {
    let mut parser = ByteParser::default();
    let mut rec = Reconstructor::default();
    rec.stats.bytes = bytes.len();
    for &b in bytes {
        if let Some(body) = parser.push(b) {
            let (seq, t, x, y, ok) = parse_body(&body);
            rec.push_frame(seq, t, x, y, ok);
        }
    }
    rec.finish();
    rec
}

pub fn write_jsonl(path: &str, recs: &[Record]) -> io::Result<()> {
    let mut f = File::create(path)?;
    for r in recs {
        writeln!(
            f,
            "{{\"seq\":{},\"t_ms\":{},\"x\":{:.5},\"y\":{:.5},\"trust\":{:.1},\"crc_ok\":{},\"held\":{}}}",
            r.seq,
            r.t_ms,
            r.x,
            r.y,
            r.trust,
            r.crc_ok,
            r.held
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc8_smbus_check_vector() {
        assert_eq!(crc8(b"123456789"), 0xF4);
    }

    fn frame(seq: u16, t_ms: u32, x_c: i16, y_c: i16) -> Vec<u8> {
        let mut b = vec![SYNC0, SYNC1];
        b.extend_from_slice(&seq.to_le_bytes());
        b.extend_from_slice(&t_ms.to_le_bytes());
        b.extend_from_slice(&x_c.to_le_bytes());
        b.extend_from_slice(&y_c.to_le_bytes());
        let c = crc8(&b[2..]);
        b.push(c);
        b
    }

    #[test]
    fn reorder_out_of_order_then_commit() {
        let mut rec = Reconstructor::default();
        let mut push = |seq: u16, x: i16| {
            let f = frame(seq, 100 * seq as u32, x, x);
            let body: [u8; 11] = f[2..].try_into().unwrap();
            let p = parse_body(&body);
            rec.push_frame(p.0, p.1, p.2, p.3, p.4);
        };
        push(0, 1000);
        push(2, 1200);
        push(1, 1100);
        rec.finish();
        let seqs: Vec<u32> = rec.out.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        assert!(rec.out.iter().all(|r| r.trust == 1.0 && !r.held));
        assert!((rec.out[1].x - 11.0).abs() < 1e-9);
    }

    #[test]
    fn bad_crc_becomes_hold_on_gap() {
        let mut bytes = frame(0, 0, 2100, 2000);
        let mut f1 = frame(1, 100, 2200, 2010);
        let last = f1.len() - 1;
        f1[last] ^= 0xFF; // break CRC; seq 1 never commits
        bytes.extend_from_slice(&f1);
        bytes.extend_from_slice(&frame(2, 200, 2300, 2020));
        let rec = reconstruct_bytes(&bytes);
        assert_eq!(rec.out.len(), 3);
        assert_eq!(rec.out[0].trust, 1.0);
        assert_eq!(rec.out[1].held, true);
        assert_eq!(rec.out[1].trust, 0.0);
        assert_eq!(rec.out[2].trust, 1.0);
        assert_eq!(rec.stats.crc_fail, 1);
    }

    #[test]
    fn hunts_sync_through_junk() {
        let mut bytes = vec![0x00, 0x11, 0xA5, 0x00];
        bytes.extend_from_slice(&frame(0, 1, 500, 500));
        let rec = reconstruct_bytes(&bytes);
        assert_eq!(rec.out.len(), 1);
        assert_eq!(rec.out[0].seq, 0);
        assert!((rec.out[0].x - 5.0).abs() < 1e-9);
    }
}
