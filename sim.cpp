#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// CRC-8/SMBUS, poly 0x07, init 0. Covers seq..payload; sync is framing only.
static uint8_t crc8(const uint8_t* p, size_t n) {
    uint8_t c = 0;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int b = 0; b < 8; b++) {
            c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
        }
    }
    return c;
}

static int16_t q100(double v) {
    double s = v * 100.0;
    if (s > 30000.0) s = 30000.0;
    if (s < -30000.0) s = -30000.0;
    return (int16_t)(s >= 0 ? s + 0.5 : s - 0.5);
}

static void put_u16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}
static void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}
static void put_i16(uint8_t* p, int16_t v) { put_u16(p, (uint16_t)v); }

// 13-byte wire frame: A5 5A | seq u16 | t_ms u32 | x i16 | y i16 | crc8
static void encode_frame(uint8_t out[13], uint16_t seq, uint32_t t_ms, int16_t x, int16_t y) {
    out[0] = 0xA5;
    out[1] = 0x5A;
    put_u16(out + 2, seq);
    put_u32(out + 4, t_ms);
    put_i16(out + 8, x);
    put_i16(out + 10, y);
    out[12] = crc8(out + 2, 10);
}

static int self_test() {
    const uint8_t* msg = (const uint8_t*)"123456789";
    uint8_t got = crc8(msg, 9);
    if (got != 0xF4) {
        fprintf(stderr, "self-test crc8 failed: got 0x%02X want 0xF4\n", got);
        return 1;
    }
    uint8_t f[13];
    encode_frame(f, 7, 1234, 2100, 1990);
    if (f[0] != 0xA5 || f[1] != 0x5A) {
        fprintf(stderr, "self-test sync failed\n");
        return 1;
    }
    if (crc8(f + 2, 10) != f[12]) {
        fprintf(stderr, "self-test frame crc mismatch\n");
        return 1;
    }
    fprintf(stderr, "sim: self-test ok crc8=0xF4 frame_bytes=13\n");
    return 0;
}

static bool load_temps(const char* path, std::vector<double>* out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char* comma = strrchr(line, ',');
        if (!comma) continue;
        char* end = nullptr;
        double v = strtod(comma + 1, &end);
        if (end == comma + 1) continue;
        out->push_back(v);
    }
    fclose(fp);
    return !out->empty();
}

struct Pending {
    uint8_t bytes[13];
    int delay;
};

static const char* arg_val(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0 && i + 1 < argc) return argv[i + 1];
    }
    return def;
}

static bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--self-test")) return self_test();

    const char* in_path = arg_val(argc, argv, "--in", "vendor/daily-min-temperatures.csv");
    const char* out_path = arg_val(argc, argv, "--out", "artifacts/capture.bin");
    const char* truth_path = arg_val(argc, argv, "--truth", "artifacts/truth.jsonl");
    uint32_t seed = (uint32_t)atoi(arg_val(argc, argv, "--seed", "1"));
    int n = atoi(arg_val(argc, argv, "--n", "4096"));
    double error_scale = atof(arg_val(argc, argv, "--error-scale", "1.0"));
    if (n < 8) n = 8;
    if (n > 65535) n = 65535;

    std::vector<double> temps;
    if (!load_temps(in_path, &temps)) {
        fprintf(stderr, "sim: failed to read temps from %s\n", in_path);
        return 1;
    }

    std::vector<double> x((size_t)n), y((size_t)n);
    const int last = (int)temps.size() - 1;
    const double two_pi = 6.283185307179586;
    double theta = 0.0;
    const double omega = two_pi / 48.0;
    for (int i = 0; i < n; i++) {
        double u = (double)i * (double)last / (double)(n - 1);
        int j = (int)u;
        if (j >= last) j = last - 1;
        double a = u - (double)j;
        double tclim = temps[(size_t)j] * (1.0 - a) + temps[(size_t)j + 1] * a;
        x[(size_t)i] = tclim + 2.0 * std::sin(theta) + 0.4 * std::sin(2.0 * theta);
        theta += omega;
        if (i == 0) y[0] = x[0];
        else y[(size_t)i] = 0.88 * y[(size_t)i - 1] + 0.12 * x[(size_t)i];
    }

    FILE* truth = fopen(truth_path, "wb");
    if (truth) {
        for (int i = 0; i < n; i++) {
            fprintf(truth, "{\"seq\":%d,\"x\":%.5f,\"y\":%.5f}\n", i, x[(size_t)i], y[(size_t)i]);
        }
        fclose(truth);
    }

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "sim: cannot write %s\n", out_path);
        return 1;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_int_distribution<int> delay_g(0, 1);
    std::uniform_int_distribution<int> delay_b(0, 4);
    std::uniform_int_distribution<int> bitpos(0, 13 * 8 - 1);
    std::uniform_int_distribution<int> junkn(1, 5);
    std::uniform_int_distribution<int> junkb(0, 255);

    // ponytail: single-link Gilbert-Elliott + delay queue; multi-hop mesh if that becomes the plant
    std::vector<Pending> pending;
    bool bad = false;
    int n_drop = 0, n_dup = 0, n_flip = 0, n_junk = 0, n_bad = 0;
    uint32_t t_ms = 0;
    int drift = 0;
    std::normal_distribution<double> N(0.0, 1.0);

    auto flush_ready = [&](bool force) {
        std::vector<Pending> keep;
        keep.reserve(pending.size());
        for (auto& p : pending) {
            if (force || p.delay <= 0) {
                fwrite(p.bytes, 1, 13, out);
            } else {
                p.delay -= 1;
                keep.push_back(p);
            }
        }
        pending.swap(keep);
    };

    const double p_bg = 0.04 * error_scale;
    const double p_gb = 0.18;
    const double p_drop_g = 0.02 * error_scale;
    const double p_drop_b = 0.35 * error_scale;
    const double p_dup_g = 0.01 * error_scale;
    const double p_dup_b = 0.08 * error_scale;
    const double p_flip_g = 0.03 * error_scale;
    const double p_flip_b = 0.55 * error_scale;
    const double p_junk_b = 0.25 * error_scale;

    for (int i = 0; i < n; i++) {
        if (bad) {
            if (U(rng) < p_gb) bad = false;
        } else {
            if (U(rng) < p_bg) bad = true;
        }
        if (bad) n_bad++;

        drift += (int)std::lround(N(rng));
        if (drift > 25) drift = 25;
        if (drift < -25) drift = -25;
        t_ms += (uint32_t)(100 + drift);

        uint8_t frame[13];
        encode_frame(frame, (uint16_t)i, t_ms, q100(x[(size_t)i]), q100(y[(size_t)i]));

        double p_drop = bad ? p_drop_b : p_drop_g;
        if (U(rng) < p_drop) {
            n_drop++;
            flush_ready(false);
            continue;
        }

        if (U(rng) < (bad ? p_flip_b : p_flip_g)) {
            int bp = bitpos(rng);
            frame[bp / 8] = (uint8_t)(frame[bp / 8] ^ (uint8_t)(1u << (bp % 8)));
            n_flip++;
        }

        if (bad && U(rng) < p_junk_b) {
            int k = junkn(rng);
            uint8_t junk[5];
            for (int j = 0; j < k; j++) junk[j] = (uint8_t)junkb(rng);
            fwrite(junk, 1, (size_t)k, out);
            n_junk++;
        }

        Pending p{};
        memcpy(p.bytes, frame, 13);
        p.delay = bad ? delay_b(rng) : delay_g(rng);
        pending.push_back(p);

        if (U(rng) < (bad ? p_dup_b : p_dup_g)) {
            Pending d = p;
            d.delay += 1 + delay_g(rng);
            pending.push_back(d);
            n_dup++;
        }
        flush_ready(false);
    }
    flush_ready(true);
    long bytes = ftell(out);
    fclose(out);

    fprintf(stderr,
            "sim: n=%d temps=%zu seed=%u error_scale=%.3f bytes_out=%ld "
            "dropped=%d flipped=%d dups=%d junk_bursts=%d ticks_bad=%d\n",
            n, temps.size(), seed, error_scale, bytes, n_drop, n_flip, n_dup, n_junk, n_bad);
    return 0;
}
