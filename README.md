# Rengin 
## A simple GPU ray tracer 
### Description:
This ray tracer is mostly based on Jamis Buck's The Ray Tracer Challenge. There is also a path tracer rendering option based on Peter Shirley's Ray Tracing in One Weekend.

It is implemented in Rust using the WGPU library and compute shaders in WGSL.
### System requirements:
- Rust and Cargo - https://www.rust-lang.org/tools/install
- A device compatible with WGPU - https://github.com/gfx-rs/wgpu

### How to run:
Pathtracer scene:
`cargo run --release -- -s assets/scenes/models.yaml -r 256 -b 16`

You can find more scenes under `assets/scenes`

The scene yaml files are built according to the format created by Jamis Buck for the Ray Tracer Challenge book (with some modifications).

### Options:
Type `cargo run --release -- -h` for a full list of options.

Key flags:
- `-s <path>` — Scene YAML file
- `-r <N>` — Rays (samples) per pixel (default: 8)
- `-b <N>` — Bounces per ray (default: 8)
- `-p` — Enable GPU profiling (per-bounce timestamps + shader stats)

### Profiling:
Run with `--profile` (`-p`) for detailed GPU performance data:
```
cargo run --release -- -s assets/scenes/models.yaml -r 8 -b 16 -p
```
This outputs per-bounce GPU timestamps (nanosecond precision via `TIMESTAMP_QUERY`) and shader-level statistics including AABB tests, BVH node visits, triangle tests, and alive ray counts.

### Bugs:
This is early days for my renderer and there are many bugs. Please let me know if you find some.

I have tested it using Vulkan and Metal backends for WGPU. I know that it currently does not work with the DX12 backend.