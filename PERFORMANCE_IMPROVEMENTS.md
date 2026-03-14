# Performance & Quality Improvements

Assessment of Rengin pathtracer against pbrt (4th edition) best practices.

## Implemented

### 1. Float Accumulation Texture (rgba8unorm → rgba16float)
- **Files**: `wgpu_utils.rs`, `pathtracer.wgsl`, `render.frag.wgsl`
- **Impact**: Enables HDR accumulation without clipping values above 1.0. Required for unbiased RR compensation and correct averaging of high-emissiveness lights.

### 2. Russian Roulette Path Termination
- **File**: `pathtracer.wgsl`
- **Formula**: `q = min(0.5, max(0, 1 - MaxComponent(throughput)))` with `depth > 1` guard (implicit: throughput=1.0 on first bounce)
- **Cap at 0.5**: Required because the MIS one-sample estimator produces extreme throughput values (light-sampled paths get throughput ~0.0001 when `v_light_pdf >> scattering_pdf`). Without the cap, 99.99% of these valuable paths are terminated.
- **Terminated paths return**: `throughput * RAY_MISS_COLOUR` (properly recorded in running average)
- **Surviving paths**: throughput scaled by `1/(1-q)` to maintain unbiased estimator
- **Note**: pbrt has no cap because their BSDF×cos/pdf is typically ≤1. Our MIS weight bakes into throughput, spanning orders of magnitude.

## Recommended (Not Yet Implemented)

### 3. Power Heuristic for MIS (Medium impact)
- Replace balance heuristic `pdf = c1*p1 + c2*p2` with power heuristic (beta=2)
- Requires tracking which strategy (light vs BSDF) generated the sample
- Attempted but reverted: the analytical light PDF used an inscribed bounding sphere radius that was too small for cube lights, zeroing out valid samples

### 4. Top-Level BVH / Scene Acceleration (High impact for complex scenes)
- `intersect()` iterates ALL objects linearly — O(N_objects) per ray
- Per-mesh BVH exists for triangle meshes but no scene-level acceleration
- Implement a TLAS/BLAS two-level structure over object bounding boxes

### 5. Microfacet BRDF / GGX (High impact — enables realistic materials)
- Current model: binary diffuse or perfect specular with linear blend
- Add GGX/Trowbridge-Reitz normal distribution with Smith masking-shadowing
- **Critical**: use visible normal distribution sampling (dramatically reduces noise for glossy)
- Half-vector Jacobian: `1 / (4 * dot(wo, wh))`

### 6. NEE for Glossy Specular (Medium impact)
- Specular paths never do direct lighting — rely on random walks hitting lights
- For glossy materials (reflective < 1.0): sample a light explicitly and weight with BSDF lobe
- Skip NEE for perfectly specular (reflective == 1.0): delta BSDF can't sample arbitrary directions

### 7. Path Regularization (Low effort, medium impact)
- After first non-specular bounce, increase roughness of subsequent BSDFs
- Double alpha, clamp to [0.1, 0.3]
- Reduces variance spikes from indirect caustics without firefly clamping

### 8. SOA Ray Buffer Layout (Medium effort, medium impact)
- Current AOS: `array<Ray>` with interleaved fields
- SOA: separate buffers for origins, directions, throughputs
- Enables coalesced memory access when all warp threads read the same field

### 9. Wavefront Kernel Decomposition (High effort, high impact)
- Current megakernel handles specular/diffuse/emissive/miss with heavy branching
- Causes warp divergence — threads hitting different materials serialize
- Separate kernels: intersection, material-sort, shade-diffuse, shade-specular
- Biggest GPU performance win for complex scenes but major refactor

### 10. Improved RNG Seeding (Low effort, low-medium impact)
- Current: seeded from ray direction float bits — correlation between similar-direction rays
- Better: pixel position + sample index + bounce depth for deterministic decorrelated sequences
- Attempted but reverted: shifting the RNG sequence changes sampling decisions, causing visible differences in noise-sensitive scenes (not a bias, just different convergence)

## Architecture Constraints

### rgba16float Accumulation
The switch from `rgba8unorm` to `rgba16float` was essential for Russian roulette. With 8-bit storage, values above 1.0 are clipped on `textureStore`, making it impossible for RR's throughput boost to compensate for terminated paths. The `rgba16float` format supports values up to 65504.

### MIS Weight in Throughput
The one-sample MIS estimator bakes `scattering_pdf / pdf` into the throughput. When `v_light_pdf >> scattering_pdf` (small distant light), this produces throughput ~0.0001. pbrt keeps MIS weights separate from path attenuation (beta), which is why their RR formula works without a cap. To remove our cap, we'd need to track path attenuation separately from MIS correction factors.

### Multi-Dispatch Bounce Architecture
Each bounce is a separate compute dispatch. The shader can't track state across bounces except through the Ray buffer. This means:
- No bounce depth counter without using a Ray struct field
- RNG is re-initialized each bounce from ray direction bits
- Russian roulette decisions are independent per-bounce
