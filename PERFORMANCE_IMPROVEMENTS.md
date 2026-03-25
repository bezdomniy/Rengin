# Performance & Quality Improvements

Assessment of Rengin pathtracer against pbrt (4th edition) best practices.

## Implemented

### 1. Float Accumulation Texture (rgba8unorm → rgba16float)
- **Files**: `wgpu_utils.rs`, `pathtracer.wgsl`, `render.frag.wgsl`
- **Impact**: Enables HDR accumulation without clipping values above 1.0. Required for unbiased RR compensation and correct averaging of high-emissiveness lights.

### 2. Russian Roulette Path Termination — DISABLED
- **File**: `pathtracer.wgsl`
- **Status**: Disabled. Throughput-based RR kills caustic paths through glass (e.g. Cornell box glass sphere). The MIS one-sample estimator bakes `scattering_pdf / pdf` into throughput, so a single diffuse bounce after a specular chain can produce very low throughput (<<0.25) even though the path is geometrically critical for the caustic. No throughput threshold can distinguish "deep attenuated path" from "low MIS weight after glass."
- **To re-enable properly**: Track explicit bounce depth (e.g. encode in `ray.pos` or spare bits) and only apply RR after depth >= 3, matching pbrt behaviour. Alternatively, separate path attenuation (beta) from MIS correction factors so RR operates on true attenuation only.
- **Path termination**: Currently handled by the bounded bounce dispatch loop.

### 3. World-Space AABB Culling (4x speedup on multi-object scenes)
- **Files**: `rt_primitives.rs`, `scene_importer.rs`, `types.wgsl`, `intersects.wgsl`
- **Impact**: Precomputed world-space AABB per object. Rays that miss an object's AABB skip the matrix transform + intersection entirely. 75% AABB skip rate on models.yaml.

### 4. Vectorized Branchless AABB Intersection (Biggest single win)
- **File**: `intersects.wgsl`
- **Impact**: Replaced scalar `for` loop with branchy swaps with vectorized `min`/`max` operations. GPUs strongly prefer vector ALU over scalar branches. ~2.5x speedup from this change alone.

### 5. BVH t-Bound Pruning
- **File**: `intersects.wgsl`
- **Impact**: AABB tests now reject nodes behind the current closest hit. Also fixed a bug where triangle comparison used the initial `inIntersection.closestT` instead of the running best `ret.closestT`. 33% speedup on bounce 1, 40% on sponza overall.

### 6. Dead Ray Elimination (Wavefront Compaction)
- **Files**: `pathtracer.wgsl`, `prepare_dispatch.wgsl`, `lib.rs`, `wgpu_utils.rs`
- **Impact**: After each bounce, alive rays write their pixel offset to a compact buffer via atomicAdd. Subsequent bounces use `dispatch_workgroups_indirect` to dispatch only alive rays. Ping-pong two index buffers for read/write isolation. First bounce keeps 2D dispatch for spatial coherence.
- **Result**: Per-sample cost now scales with alive rays. Table scene: 28fps → 142fps on later samples (5x). Saves ~4ms/sample on models.yaml.

### 7. GPU Timestamp Profiling
- **Files**: `wgpu_utils.rs`, `lib.rs`
- **Impact**: `--profile` (`-p`) flag enables `TIMESTAMP_QUERY` for nanosecond per-bounce GPU timing + shader-side atomic counters (AABB tests, BVH nodes, triangles, alive rays). Single command encoder submission for raygen + pathtracer + render.

### Cumulative Performance Results

**models.yaml** (1600x1200, 3 meshes totaling 177k tris, 16 bounces):

| Version | GPU Time/Sample |
|---------|----------------|
| Original | ~80ms |
| + AABB culling + vectorized intersection | ~26ms |
| + BVH t-bound pruning | ~20ms |
| + Dead ray elimination | ~18ms (avg across bounces) |

**sponza.yaml** (1800x1200, 570k tris, 16 bounces): ~155ms → ~92ms

## Recommended (Not Yet Implemented)

### Power Heuristic for MIS (Medium impact)
- Replace balance heuristic `pdf = c1*p1 + c2*p2` with power heuristic (beta=2)
- Requires tracking which strategy (light vs BSDF) generated the sample

### Microfacet BRDF / GGX (High impact — enables realistic materials)
- Current model: binary diffuse or perfect specular with linear blend
- Add GGX/Trowbridge-Reitz normal distribution with Smith masking-shadowing
- **Critical**: use visible normal distribution sampling (dramatically reduces noise for glossy)

### NEE for Glossy Specular (Medium impact)
- Specular paths never do direct lighting — rely on random walks hitting lights
- For glossy materials (reflective < 1.0): sample a light explicitly and weight with BSDF lobe

### Wider BVH Nodes (Medium effort, high impact for sponza-class scenes)
- Current binary BVH: 128 node visits per ray on sponza (570k tris)
- 4-wide or 8-wide BVH nodes reduce tree depth and improve SIMD utilization
- Would cut sponza BVH traversal by ~40-50%

### Wavefront Material Sorting (High effort, high impact)
- Current megakernel handles specular/diffuse/emissive/miss with heavy branching
- Causes warp divergence — threads hitting different materials serialize
- Sort rays by material type between intersection and shading phases

## Architecture Constraints

### rgba16float Accumulation
The switch from `rgba8unorm` to `rgba16float` supports HDR values above 1.0 for correct averaging of high-emissiveness lights. Also required if RR is re-enabled (surviving paths get throughput boost above 1.0).

### MIS Weight in Throughput
The one-sample MIS estimator bakes `scattering_pdf / pdf` into the throughput. When `v_light_pdf >> scattering_pdf` (small distant light), this produces throughput ~0.0001. pbrt keeps MIS weights separate from path attenuation (beta). This conflation is why throughput-based RR doesn't work correctly — it cannot distinguish low attenuation from low MIS weight. To fix, track path attenuation separately from MIS correction factors.

### Wavefront Bounce Architecture
Each bounce is a separate compute dispatch with active ray compaction between bounces. Alive rays write to a compact index buffer, and the next bounce dispatches only those rays via indirect dispatch. The shader can't track state across bounces except through the Ray buffer and compaction buffers.
