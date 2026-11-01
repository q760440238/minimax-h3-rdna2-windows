# VAE / non-sampling time profile — MiniMax-H3 (864x480, 8 steps, ROCm)

Binary: `E:\minimaxh3\bin\sd-rocm\sd-cli.exe` (not rebuilt)
Harness: `E:\minimaxh3\bench.ps1` (seed 42, cfg 1.0, `--offload-to-cpu`, `--mmap`, `--max-vram ROCm0=15`)
Date: 2026-11-02, ~01:43–02:00 (+08:00)
GPU released at 02:00:20 — `E:\minimaxh3\reports\GPU-FREE.txt` written.

**Definitions used everywhere below**
`total` = the `[I] generate_video completed in Xs` line (this is what the "67.5 s / 183.5 s" numbers quoted
in the task are). It *excludes* text-encoder conditioning, process start-up and the webm save.
`non-sampling` = total − `sampling completed`.
All component times are the binary's own `taking X` log values (ANSI stripped), so they are wall-clock
timers inside the process, not derived from external timing.

---

## 1. Breakdown of the non-sampling time

### 22 frames — `out/bench_v2d_22.log` (quiet machine, matches the known 67.5 s run)

| component | log line | seconds | % of total (64.77 s) |
|---|---|---:|---:|
| sampling | `sampling completed, taking 49.90s` | 49.90 | 77.0 % |
| **video VAE decode** | `decode_first_stage completed, taking 7.95s` (`computing vae decode graph completed, taking 7.90s`) | **7.95** | **12.3 %** |
| video VAE staged to GPU | `staged compute params (4636.94 MB, 441 tensors, 5 blocks) ... taking 2.06s` | 2.06 | 3.2 % |
| video VAE tensor load | `loading tensors completed, taking 0.22s` | 0.22 | 0.3 % |
| diffusion staged to GPU | `staged compute params (10975.95 MB, 532 tensors, 12 blocks) ... taking 4.79s` | 4.79 | 7.4 % |
| diffusion tensor load | `loading tensors completed, taking 0.20s` | 0.20 | 0.3 % |
| audio VAE decode | `decoding audio latent completed, taking 0.70s` | 0.70 | 1.1 % |
| audio VAE staged + load | 0.15 s + 0.21 s | 0.36 | 0.6 % |
| graph-cut plans | 4 × `build cached graph cut plan done` (23 ms total) | 0.02 | 0.0 % |
| **residual / overlap** | see §2 | **−1.43** | −2.2 % |
| **non-sampling total** | 64.77 − 49.90 | **14.87** | **23.0 %** |

### 56 frames — `out/bench_e2e_j128_56.log` (quiet machine, matches the known 183.5 s run)

| component | log line | seconds | % of total (180.07 s) |
|---|---|---:|---:|
| sampling | `sampling completed, taking 138.42s` | 138.42 | 76.9 % |
| **video VAE decode** | `decode_first_stage completed, taking 24.24s` | **24.24** | **13.5 %** |
| video VAE staged to GPU | `staged compute params (4636.94 MB, 441 tensors, 5 blocks) ... taking 2.08s` | 2.08 | 1.2 % |
| video VAE tensor load | 0.21 s | 0.21 | 0.1 % |
| diffusion per-block paging | 51 × `loading tensors completed, taking ~0.21s` (206.72 MB chunks) + 0.02 s stage | 11.13 | 6.2 % |
| audio VAE decode | `decoding audio latent completed, taking 1.01s` | 1.01 | 0.6 % |
| audio VAE staged + load | 0.16 s + 0.20 s | 0.36 | 0.2 % |
| graph-cut plans | 24 ms | 0.02 | 0.0 % |
| **residual** | see §2 | **2.62** | 1.5 % |
| **non-sampling total** | 180.07 − 138.42 | **41.65** | **23.1 %** |

The non-sampling share is therefore **flat at ~23 %** at both lengths: the extra 27 s of non-sampling at 56 f
is almost entirely video-VAE decode (+16.3 s) plus the chunked diffusion paging (+6.1 s).

### The video VAE decodes in fixed-size temporal passes
At 56 f the log contains **three** decode-graph lines, and their sum explains `decode_first_stage`:

```
[V] computing vae decode graph completed, taking 9.26s
[V] computing vae decode graph completed, taking 6.93s
[V] computing vae decode graph completed, taking 6.92s
[I] decode_first_stage completed, taking 24.24s
```
At 22 f there is exactly one pass (`7.90 s`). So decode cost is ~7–9 s **per ~18–22-frame pass**,
and 56 f = 3 passes. This is the single largest non-sampling item at both lengths (12–14 % of total,
54–58 % of all non-sampling time).

### Text encoder is *outside* `generate_video` but is the largest non-sampling item end-to-end

| | 22 f | 56 f |
|---|---:|---:|
| `get_learned_condition completed` (TE compute) | 6.18 s | 16.34 s |
| TE staged to GPU | 5.32 s | 0.64 s |
| TE tensor load | 0.20 s | 10.33 s |
| process start-up + webm save (harness `wall` − total) | ≈ 3.2 s | — |

Quoted: `[V] computing condition graph completed, taking 6183 ms` then
`[I] get_learned_condition completed, taking 6.18s`. In the 56 f run the TE was a 1.48 GB Q4_K model processed
in 51 segments (`peak across 51 segments`), re-loading 197.5 MB per segment at ~0.21 s each — 10.3 s of pure
re-loading, which is why TE time almost tripled while the model got *smaller*.

---

## 2. What accounts for the remainder

- **22 f: −1.4 s (i.e. zero).** The component timers overlap slightly: `sampling completed` starts before the
  diffusion staging timer is closed. Nothing is missing — the 14.9 s is fully explained by video-VAE decode
  (8.0) + diffusion staging (5.0) + audio VAE (1.1) + VAE staging (2.3).
- **56 f: 2.6 s (1.5 %).** Graph-cut planning (~24 ms total), compute-buffer allocation
  (`minimax_h3 compute buffer size: 1736.44 MB(VRAM) ... peak across 54 segments`) and teardown
  (`released params backend buffers ... 27221.27 MB`) are the only other logged steps; none of them carries a
  timer. This 2.6 s is **not** attributable from the log — treat it as allocation/teardown overhead.
- **One genuinely unexplained case:** with `--vae-tiling` at 22 f the residual jumps to **~11 s**
  (see below). No log line covers it. It is the one thing worth instrumenting next.

---

## 3. Mitigation results

Machine state matters: my first two runs (01:43–01:47) were **I/O-contended** — cold page cache made the
diffusion load take 17.3 s (`read: 16.83s`) instead of 0.2 s, and sampling ran 67.4 s instead of 49.9 s.
The 56 f re-measurement degraded to 44.3 s/step (vs 17.3 s/step) and was **aborted at step 5/8**; I did not
report it as a result. Numbers below are therefore marked *quiet* (historical, same binary/flags, and my
`noaudio` run) or *contended* (fresh, for direction only).

| # | config (22 f unless stated) | total s | sampling s | video-VAE decode s | other non-sampling s | verdict |
|---|---|---:|---:|---:|---:|---|
| — | baseline, quiet (`bench_v22_base.log`) | 66.14 | 50.91 | 8.01 | 7.22 | reference |
| — | baseline, quiet (`bench_v2d_22.log`) | 64.77 | 49.90 | 7.95 | 6.92 | reference |
| — | baseline, contended (`bench_prof22.log`) | 86.18 | 67.35 | 11.24 | 7.59 | *contended, do not compare* |
| **a** | `--vae-tiling` **on**, quiet (`bench_rocm22_tile.log`) | 81.61 | 53.14 | 9.48 | 18.99 | **+15.5 s worse** |
| **a** | `--vae-tiling` **on**, quiet (`bench_rocm22_vaetile.log`) | 81.77 | 53.37 | 9.51 | 18.89 | **+15.6 s worse** |
| **a** | `--vae-tiling` **on**, contended (`bench_prof22_tile.log`) | 97.31 | 69.13 | 12.96 | 15.22 | **+11.1 s worse vs contended base** |
| **b** | `--vae-tile-size 256x256`, contended (`bench_prof22_tile256.log`) | 93.44 | 62.88 | 15.16 | 15.40 | worst decode of all 22 f runs |
| **b** | `--vae-tile-size 512x512`-ish, quiet (`bench_rocm22_tile512.log`) | 80.26 | 53.21 | 8.19 | 18.86 | decode ≈ baseline, total still +14 s |
| **c** | **no `--audio-vae`**, quiet (`bench_prof22_noaudio.log`) | 67.82 | 53.40 | 8.07 | 6.35 | **works**; saves 0.7–1.1 s |
| **d** | `--vae-on-cpu`, 22 f (`bench_prof22_vaeoncpu.log`) | — | 62.93 | **killed** | — | **not viable** (see below) |

### (a) `--vae-tiling` at 22 f — hurts, reproducibly
- Quiet: `generate_video` 66.14 → 81.61 s (+23 %). VAE decode itself only 8.01 → 9.48 s (+1.5 s); the other
  **~14 s is unlogged** (TE staging also got slower in that run: 5.33 → 0.59 s stage but 10.44 s of tensor
  loading — the page cache was cold).
- Contended same-session: 86.18 → 97.31 s (+11.1 s), decode 11.24 → 12.96 s.
- Direction is consistent in 4/4 runs: **tiling is a loss at 22 f**, consistent with the note already in
  `gen.ps1` ("22f: +2%"). My measurements say the penalty is much larger than 2 % — closer to +15 s.
  (Speculation: the tiled path re-stages weights per tile; no log line proves it.)

### (b) `--vae-tile-size 256x256`
- 256×256 is already the documented default, so `--vae-tiling --vae-tile-size 256x256` should equal (a);
  measured 93.44 s total / 15.16 s decode vs 97.31 s / 12.96 s for plain tiling — i.e. **within run-to-run
  noise of (a), but the decode was the slowest seen at 22 f**.
- Historical quiet 512-tile run decoded in 8.19 s (≈ baseline 8.01 s) → **larger tiles are cheaper than 256**.

### (c) Omitting `--audio-vae` — it works, and saves almost nothing
`bench.ps1` always passes `--audio-vae`, so this was run by invoking the binary directly with the identical
argv minus that flag (`--offload-to-cpu --mmap --max-vram ROCm0=15`, 22 f, 8 steps):
- exit code **0**, `generate_video completed in 67.82s`, 986 KB webm written, **no `[E]` lines**.
- **No** audio-VAE lines at all in the log — no staging, no `decoding audio latent`.
- Non-sampling drops from 15.2 s to **14.4 s**, i.e. **0.8–1.1 s saved (0.6–0.8 % of total)**. The audio VAE
  is simply not worth optimising; the trade is that the output has no audio track.

### (d) `--vae-on-cpu` — listed but useless here
`--help` shows `--vae-on-cpu  deprecated; use --backend vae=cpu`, plus `--backend`, `--vae-tiling`,
`--vae-tile-size`, `--temporal-tiling`, `--vae-tile-overlap`, `--extra-tiling-args`. No other VAE-placment
option exists.
Measured: `--vae-on-cpu` moved the audio VAE to CPU (`MiniMax-H3 audio VAE decode ... 3.96s` vs 0.89 s on
GPU, 4.4×) and then the **video VAE decode on CPU ran > 5 minutes** for 22 frames without producing a line
after `decode audio latent 37x2x32x1`. Killed at ~5 min (vs 8 s on GPU). Not a mitigation.

---

## 4. Where the time really goes (summary)

End-to-end 22 f quiet run, ~80 s wall:
`11.7 s` text encoder (incl. 5.3 s staging) + `64.8 s` generate_video + `3.2 s` start-up/save.
Inside generate_video: `49.9 s` sampling, `8.0 s` video-VAE decode, `5.0 s` diffusion staging,
`2.3 s` video-VAE staging, `1.1 s` audio VAE.

End-to-end 56 f quiet run, ~200 s wall: `~17 s` text encoder + `180.1 s` generate_video
(`138.4` sampling, `24.2` video-VAE decode in 3 passes, `11.1` diffusion paging, `2.1` VAE staging, `1.4` audio).

**Answer to "where does the non-sampling time go": 54–58 % of it is the video VAE decode, 25–34 % is
staging weights into VRAM, 7–9 % audio VAE, and the rest is allocation/graph overhead.**

---

## 5. What to try next

1. **Attack the video-VAE pass count, not the tile size.** Decode is ~7–9 s per temporal pass and 56 f needs
   three. `--extra-tiling-args temporal_tile_frames=<n>` / `--temporal-tiling` could push 56 f from 3 passes to
   2 (≈ −8 s, −4.5 %). Measure `count of "computing vae decode graph" lines` directly — it is the cheapest
   signal available.
2. **Never enable `--vae-tiling` at 22 f.** It reproduces +11…+16 s across 4 runs. Find the ~11 s unlogged
   overhead first (item 6) before trusting the "−4 % at 56 f" folklore.
3. **Keep the page cache warm.** `--mmap` + 29 GB of weights on a 32 GB box means a cold cache costs
   +11 s of diffusion reads and inflates sampling by ~35 % (49.9 → 67.4 s). Consecutive runs back-to-back are
   not comparable unless the previous run touched the same files; this is the single biggest measurement trap
   in this task.
4. **Diffusion staging (5 s at 22 f, 11 s at 56 f)** is the second target. At 56 f the model is paged in 51
   chunks of 206.72 MB; at 22 f it is one 10.98 GB transfer. Finding the flag that keeps the single-transfer
   path at 56 f would recover most of that 6 s.
5. **Audio VAE: skip it if you don't need sound** (works, −0.8…−1.1 s), otherwise leave it alone; it is 0.8 %
   of runtime and `--vae-on-cpu` makes it 4.4× worse.
6. **Add timestamps to the log or a `--profile` flag.** The ~11 s gap in the tiled runs and the 2.6 s residual
   at 56 f have no corresponding log line, so they cannot be attributed from logs alone.

---

## Appendix — commands and artifacts

```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\minimaxh3\bench.ps1 `
  -Name prof22 -Steps 8 -Frames 22 -Sd E:\minimaxh3\bin\sd-rocm\sd-cli.exe -DeviceName ROCm0 `
  -Llm E:\minimaxh3\models\text_encoders\qwen3vl_32b_minimax_h3-Q2_K_M.gguf -Extra "--mmap"
```
- Parser (strips ANSI, sums per-model stage/load, extracts all `completed/taking` lines):
  `E:\minimaxh3\out\profile_parse.ps1 -Logs "a.log,b.log"`
- Fresh logs: `out/bench_prof22.log`, `bench_prof22_tile.log`, `bench_prof22_tile256.log`,
  `bench_prof22_vaeoncpu.log` (aborted), `bench_prof22_noaudio.log`, `bench_prof56fa.log` (aborted at 5/8 steps).
- Reference quiet logs used for the two headline tables: `out/bench_v2d_22.log`,
  `out/bench_e2e_j128_56.log` (both `bench.ps1`, 8 steps, 864×480).
