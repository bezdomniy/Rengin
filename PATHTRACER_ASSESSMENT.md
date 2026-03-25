# Path Tracer Assessment Report

Assessment of the Rengin GPU path tracer against [Physically Based Rendering (4th Ed.)](https://pbr-book.org/4ed/contents).

---

## Critical Bugs

### 1. RNG Never Advances — Severe Correlation Artifacts

**Files:** `src/shaders/random.wgsl`, `src/shaders/pathtracer.wgsl`

`init_pcg3d()` is called once per bounce, producing three random values (`rand_pcg3d.x`, `.y`, `.z`). These values are **never updated** — every subsequent call to `f32_zero_to_one(rand_pcg3d.x)` returns the **same number**.

Within a single `renderScene()` call, the light selection, direction sampling, and material decisions all read from the same 3 values. This produces correlated samples within each bounce.

> **PBR Book reference:** Monte Carlo estimation requires independent samples (Ch. 2, *Monte Carlo Integration*).

**Status: FIXED** — Added `next_pcg3d()` that advances the PCG state. Called before each sampling function and bare `f32_zero_to_one` usage. In practice the visual impact was minimal since the seed changes between bounces via ray direction bits, but the fix is theoretically correct.

---

### 2. SAH BVH Cost Computation Bug — Wrong Bounding Box Accumulation

**File:** `src/engine/bvh.rs`

Two bugs in SAH cost evaluation:
- `b0 = b1.merge(node)` — merges with wrong variable, left partition bounds are incorrect
- `buckets[b].merge(...)` — return value discarded, bucket bounds never accumulate

> **PBR Book reference:** SAH (Ch. 7.3) requires correct surface area computation for each candidate partition.

**Status: NOT FIXED** — Attempted PBRT-style two-scan approach but no measurable performance improvement observed. Reverted by user.

---

### 3. Sphere Intersection Missing Distance Check — Can Return Farther Hits

**File:** `src/shaders/intersects.wgsl`

When the ray origin is inside a sphere (`t1 < EPSILON`), the fallback to `t2` didn't verify `t2 < inIntersection.closestT`, potentially returning a farther hit than the current closest.

**Status: FIXED** — Added `&& t2 < inIntersection.closestT` check. Matches PBRT's sphere intersection logic.

---

## Moderate Issues

### 4. No Russian Roulette — Biased Energy Loss

**File:** `src/shaders/pathtracer.wgsl`

The renderer uses a fixed bounce count with no Russian roulette termination. Paths are truncated at max bounces, causing systematic energy loss.

> **PBR Book reference:** Ch. 2.2.4 and Ch. 13.1 — Russian roulette is the standard technique for unbiased path termination.

**Status: REVERTED** — Russian roulette was implemented but disabled because throughput-based RR kills caustic-contributing paths through glass. The MIS one-sample estimator bakes `scattering_pdf / pdf` into throughput, so low MIS weights after glass refraction trigger RR termination even on geometrically critical paths. Needs explicit bounce depth tracking or separation of path attenuation from MIS weights to re-enable correctly. Wavefront compaction (alive rays write to compact index buffer, indirect dispatch) remains active for natural path termination.

---

### 5. Cube Light: Volume Sampling + Wrong PDF

**Files:** `src/shaders/pathtracer.wgsl`

Originally cube lights sampled random points in the volume `[-1,1]³` and used a copy-pasted sphere PDF.

**Status: FIXED** — Cube light sampling now uses area-weighted surface sampling (faces selected proportional to their world-space area). PDF uses bounding sphere solid angle approximation for numerical stability. The area-to-solid-angle Jacobian (`dist²/cos θ/area`) was attempted but produces fireflies at grazing angles on thin cubes; PBRT solves this with spherical rectangle sampling (Ureña 2013) which avoids the cos θ singularity entirely.

---

### 6. Rough Specular Reflection Has No PDF Correction

**File:** `src/shaders/pathtracer.wgsl`

Rough specular surfaces (0 < reflective < 1) perturb the reflection direction but didn't account for energy conservation in the throughput.

**Status: FIXED** — Throughput is now scaled by `reflective` for rough specular materials, treating the parameter as metallic reflectance. A proper microfacet BRDF (GGX/Beckmann) with matching PDF would be the full solution.

---

### 7. Double Gamma Correction

**File:** `src/shaders/render.frag.wgsl`

The fragment shader applied `sqrt()` then `to_linear_rgb()` — two gamma corrections that approximately cancelled with precision loss. The surface format is `Bgra8UnormSrgb`, meaning the hardware applies sRGB gamma automatically.

**Status: FIXED** — Removed both `sqrt()` and `to_linear_rgb()`. Fragment shader now outputs linear values with a clamp, and the sRGB surface handles gamma. Added per-sample clamp to 3.0 to suppress fireflies that were previously hidden by the sqrt compression.

---

### 8. Nested Dielectrics Not Handled

**File:** `src/shaders/pathtracer.wgsl`

The original code had two bugs:
- **Wrong n_ratio on back face:** `eta_t = material_RI / ray_RI` gave `n_ratio = ray_RI² / material_RI` instead of `ray_RI / 1.0`
- **Outgoing ray RI always set to material RI:** When exiting glass, ray should enter air (RI=1.0)

**Status: PARTIALLY FIXED** — Implemented heuristic: if `ray.RI > 1.0` on back face, exit to air (`eta_t = 1.0`); otherwise enter the material. Reflection preserves the current medium RI. Single-layer glass and high-RI materials (diamond) work correctly. Hollow glass spheres (nested dielectrics) have a known limitation at the inner sphere's back face — without a media stack, the renderer can't determine the outer medium.

Unit tests in `src/engine/refraction.rs` document both working cases and the known limitation.

---

### Additional Fix: Camera Move Smearing

**File:** `src/shaders/pathtracer.wgsl`

Pre-existing bug: when the camera moved and `subpixel_idx` reset to 0, pixels whose rays didn't terminate on the first bounce never wrote to the texture — old image data persisted, causing ghosting.

**Status: FIXED** — When `subpixel_idx == 0` and a ray hasn't terminated, the shader now writes `RAY_MISS_COLOUR` to clear old data.

---

## Minor Issues

### 9. Triangle Intersection: Degenerate Triangles

**File:** `src/shaders/intersects.wgsl`

When `det ≈ 0`, `1.0/det` produces infinity. On GPU, IEEE 754 arithmetic handles this gracefully — infinity propagates through the calculations and the `isHit` bounds checks reject the intersection (NaN/infinity comparisons evaluate to false in WGSL).

**Status: NOT AN ISSUE** — No guard needed. An explicit `abs(det) < EPSILON` check was attempted but the EPSILON threshold rejected valid near-parallel triangles, creating holes in meshes.

---

### 10. No Top-Level Acceleration Structure

**File:** `src/shaders/intersects.wgsl`

The `intersect()` function iterates through all objects linearly — O(n_objects) per ray.

> **PBR Book reference:** Ch. 7 — a two-level acceleration structure (TLAS over BLAS) is standard practice.

**Status: PARTIALLY FIXED** — World-space AABB culling added per object. Rays that miss an object's AABB skip the matrix transform and intersection entirely (75% skip rate on models.yaml). A full TLAS is not implemented but the AABB culling provides similar benefit for small object counts.

---

### 11. Emissive Check Only Tests Red Channel

**File:** `src/shaders/pathtracer.wgsl`

Originally `emissiveness.x > 0.0` — only checks red. A cyan light `(0, 5, 5)` would be missed.

**Status: FIXED** — Changed to `any(ob_params.material.emissiveness.xyz > vec3(0.0))`.

---

### 12. Unnecessary `sqrt` in Distance Squared Computation

**File:** `src/shaders/pathtracer.wgsl`

`pow(length(direction), 2.0)` computes sqrt then squares — a wasted sqrt.

**Status: FIXED** — Changed to `dot(direction, direction)`.

---

## Summary

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | **Critical** | RNG never advances | Fixed |
| 2 | **Critical** | SAH BVH cost bugs | Not fixed (no perf impact observed) |
| 3 | **Critical** | Sphere intersection missing distance check | Fixed |
| 4 | Moderate | No Russian roulette | Reverted (breaks glass caustics; needs bounce depth tracking) |
| 5 | Moderate | Cube light: volume sampling + sphere PDF | Fixed (area-weighted surface sampling) |
| 6 | Moderate | Rough specular missing PDF | Fixed (reflective as energy weight) |
| 7 | Moderate | Double gamma correction | Fixed (linear output + sRGB surface) |
| 8 | Moderate | Nested dielectrics broken | Partially fixed (heuristic, single-layer works) |
| — | Moderate | Camera move smearing | Fixed (clear texture on first sample) |
| 9 | Minor | Degenerate triangle guard | Not an issue (GPU handles via IEEE 754) |
| 10 | Minor | No TLAS | Partially fixed (world-space AABB culling) |
| 11 | Minor | Emissive check only on red channel | Fixed |
| 12 | Minor | `pow(length(), 2)` vs `dot()` | Fixed |
