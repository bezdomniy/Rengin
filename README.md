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

Unknown keys, shape, material and transform names are reported as errors when a scene loads. Meshes are added with `add: obj` and a `file:` path relative to the scene file. Patterns, `args` and `shadow` are accepted but not rendered yet.

### Options:
Type `cargo run --release -- -h` for a full list of options.

Key flags:
- `-s <path>` — Scene YAML file
- `-r <N>` — Rays (samples) per pixel (default: 8)
- `-b <N>` — Bounces per ray (default: 8)
- `--render-scale <N>` — Scale the scene camera's render resolution without changing window size (default: 1)
- `-p`, `--profile` — Enable low-overhead per-bounce GPU timestamps
- `--profile-stats` — Also collect instrumented shader counters (implies `--profile`)

### Profiling:
Run with `--profile` (`-p`) for per-bounce GPU timestamps:
```
cargo run --release -- -s assets/scenes/models.yaml -r 8 -b 16 -p
```
Timestamp readback is asynchronous, and the normal rendering path does not wait for each GPU submission to finish. To diagnose traversal work as well, use `--profile-stats`; it adds shader-level counters for AABB tests, BVH node visits, triangle tests, and active rays, but its instrumentation affects timings.

The camera width and height in the scene YAML define the internal render resolution. Window resizing only changes presentation, preserving the camera aspect ratio. On the web, `render_scale` can also be supplied as a URL parameter.

### Running in the browser:
The web build uses WebGPU and is built with [trunk](https://trunkrs.dev):
```
trunk serve --release
```
Options are URL parameters, e.g. `?scene=glass_test&rpp=256&bounces=8&render_scale=1`. Only the scenes embedded in `src/engine/embedded_assets.rs` are available (`cornellbox`, `cornellbox2`, `fresnel`, `glass_test`), and OBJ models cannot be loaded yet.

Pushing to `master` deploys the web build to GitHub Pages at `/Rengin/` (`.github/workflows/pages.yml`).

### Bugs:
This is early days for my renderer and there are many bugs. Please let me know if you find some.

I have tested it using Vulkan and Metal backends for WGPU. I know that it currently does not work with the DX12 backend.
