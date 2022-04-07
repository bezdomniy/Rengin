# Rengin 
## A simple GPU ray tracer 
### Description:
This ray tracer is mostly based on Jamis Buck's The Ray Tracer Challenge. There is also a path tracer rendering option based on Peter Shirley's Ray Tracing in One Weekend (use option -p).

It is implemented in Rust using the WGPU library and compute shaders in WGSL.
### System requirements:
- Rust and Cargo - https://www.rust-lang.org/tools/install
- A device compatible with WGPU - https://github.com/gfx-rs/wgpu 

### How to run:
Raytracer scene:
`cargo run --release -- -s assets/scenes/reflectionScene.yaml`

Pathtracer scene:
`cargo run --release -- -s assets/scenes/models.yaml -p -r 256`

You can find more scenes under `assets/scenes`

The scene yaml files are built according to the format created by Jamis Buck for the Ray Tracer Challenge book (with some modifications).

### Options:
Type `cargo run --release -- -h` for a full list of options.

### Bugs:
This is early days for my renderer and there are many bugs. Please let me know if you find some.

I have tested it using Vulkan and Metal backends for WGPU. I know that it currently does not work with the DX12 backend.