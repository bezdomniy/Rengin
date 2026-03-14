# Path Tracer Assessment Report

Assessment of the Rengin GPU path tracer against [Physically Based Rendering (4th Ed.)](https://pbr-book.org/4ed/contents).

---

## Critical Bugs

### 1. RNG Never Advances — Severe Correlation Artifacts

**Files:** `src/shaders/random.wgsl`, `src/shaders/pathtracer.wgsl:271`

`init_pcg3d()` is called once per bounce, producing three random values (`rand_pcg3d.x`, `.y`, `.z`). These values are **never updated** — every subsequent call to `f32_zero_to_one(rand_pcg3d.x)` returns the **same number**.

Within a single `renderScene()` call:

| Call site | Component used |
|-----------|---------------|
| `random_light()` (line 89) | `rand_pcg3d.x` |
| `random_to_sphere()` (line 144) | `rand_pcg3d.x`, `.y` |
| `random_cosine_direction()` (line 132) | `rand_pcg3d.x`, `.y` |
| scatter/light decision (line 210) | `rand_pcg3d.y` |
| Fresnel probability (line 191) | `rand_pcg3d.y` |

There are effectively **3 random numbers per bounce, reused across all decisions**. The light selection, direction sampling, and material decisions are all correlated. This produces structured noise patterns instead of the white noise expected from Monte Carlo integration.

> **PBR Book reference:** Monte Carlo estimation requires independent samples (Ch. 2, *Monte Carlo Integration*). Correlated samples violate this assumption and can create systematic bias patterns that don't converge correctly.

**Fix:** Add a step function that advances the PCG state between draws:

```wgsl
fn next_pcg3d() {
    rand_pcg3d = rand_pcg3d * 1664525u + 1013904223u;
    rand_pcg3d.x += rand_pcg3d.y * rand_pcg3d.z;
    rand_pcg3d.y += rand_pcg3d.z * rand_pcg3d.x;
    rand_pcg3d.z += rand_pcg3d.x * rand_pcg3d.y;
    rand_pcg3d ^= rand_pcg3d >> vec3<u32>(16u);
}
```

Call `next_pcg3d()` before each sampling function.

---

### 2. SAH BVH Cost Computation Bug — Wrong Bounding Box Accumulation

**File:** `src/engine/bvh.rs:399-402`

```rust
for node in buckets.iter().take(i + 1) {
    b0 = b1.merge(node);   // BUG: should be b0.merge(node)
    count0 += node.skip_ptr_or_prim_idx1;
}
```

`b0` (the left partition bounds) is computed as `b1.merge(node)`, but `b1` is never modified in this loop — it stays as `NodeInner::empty()`. So `b0` ends up being just the **last bucket's bounds**, not the union of all left-side buckets. The SAH cost estimates are wrong, leading to suboptimal split decisions.

> **PBR Book reference:** SAH (Ch. 7.3) requires computing the surface area of bounding boxes enclosing each candidate partition. Incorrect bounds → incorrect cost → poor tree quality → slower traversal.

**Fix:** Change line 400 to `b0 = b0.merge(node);`

---

### 3. Sphere Intersection Missing Distance Check — Can Return Farther Hits

**File:** `src/shaders/intersects.wgsl:111-119`

```wgsl
if t1 < inIntersection.closestT || t2 < inIntersection.closestT {
    if t1 < t2 && t1 > EPSILON {
        return Intersection(..., t1, ...);
    }
    if t2 > EPSILON {
        return Intersection(..., t2, ...);  // BUG: no check t2 < inIntersection.closestT
    }
}
```

When the ray origin is inside a sphere (`t1 < EPSILON`), the preliminary guard passes because `t1 < inIntersection.closestT` (t1 is negative). The code then falls through to return `t2` with only a `t2 > EPSILON` check, but **never verifies `t2 < inIntersection.closestT`**. This can return a hit that is farther than the current closest, causing incorrect z-ordering for rays inside glass spheres.

**Fix:** Add `&& t2 < inIntersection.closestT` to line 116.

---

## Moderate Issues

### 4. No Russian Roulette — Biased Energy Loss

**File:** `src/shaders/pathtracer.wgsl`

The renderer uses a fixed bounce count (`ray_bounces`) with no Russian roulette termination. This introduces **systematic bias** — paths that would contribute energy beyond the max bounce count are simply truncated. Russian roulette provides an **unbiased** termination strategy: with probability `q`, terminate and scale surviving paths by `1/(1-q)`.

> **PBR Book reference:** Ch. 2.2.4 and Ch. 13.1 — Russian roulette is the standard technique for unbiased path termination. The estimator `E[F_rr] = q·0 + (1-q)·F/(1-q) = E[F]` is unbiased by construction.

**Impact:** Energy loss at every bounce cutoff. Scenes with many inter-reflections (e.g., glass inside a Cornell box) will appear darker than ground truth.

**Fix:** After each bounce, compute continuation probability based on throughput:

```wgsl
let p_continue = max(ray.throughput.r, max(ray.throughput.g, ray.throughput.b));
if f32_zero_to_one(rand_pcg3d.z) > p_continue {
    // terminate
    rays[offset] = Ray(vec3(-1f), -1f, vec3(-1f), 0u, vec4(-1f));
    return vec4(-1f);
}
ray.throughput /= p_continue;
```

---

### 5. Cube Light: Volume Sampling + Wrong PDF — Biased Estimator

**Files:** `src/shaders/pathtracer.wgsl:110-114` and `src/shaders/pathtracer.wgsl:135-142`

Two compounding errors for cube lights:

**a) Sampling the volume, not the surface (line 111):**

```wgsl
let p = random_in_cube();  // samples [-1,1]^3 VOLUME
```

A surface light should sample points on its **surface**. Sampling the interior produces directions that don't correspond to actual surface points.

**b) PDF uses sphere approximation (lines 135-142):**

The cube `light_pdf` is an exact copy of the sphere PDF (bounding sphere solid angle). The PDF doesn't match the actual sampling distribution, violating the fundamental requirement of Monte Carlo estimation: `E[f/p] = ∫f` only when `p` is the actual distribution of the samples.

> **PBR Book reference:** Ch. 12.5 — Light sampling PDFs must match the actual sampling distribution. A PDF/sampling mismatch produces a **biased** estimator.

**Impact:** Cube lights will have incorrect brightness and falloff.

---

### 6. Rough Specular Reflection Has No PDF Correction

**File:** `src/shaders/pathtracer.wgsl:174-175`

```wgsl
direction = hitParams.reflectv
    + ((1.0 - ob_params.material.reflective) * onb * random_cosine_direction());
```

The reflected direction is perturbed by a random cosine-weighted offset scaled by `(1 - reflective)`, but the throughput is simply `throughput * colour` with no PDF or BRDF evaluation for this glossy distribution.

> **PBR Book reference:** Ch. 9 and Ch. 13 — any non-delta BRDF bounce must compute the throughput weight as `f(wo,wi) * |cos θ| / pdf(wi)`. The missing PDF means rough mirrors will have incorrect brightness and energy is not conserved.

---

### 7. Double Gamma Correction

**File:** `src/shaders/render.frag.wgsl:22-26`

```wgsl
color = sqrt(color);                    // gamma 2.0 correction
color = clamp(color, ...);
return to_linear_rgb(color);            // sRGB transfer function (power 2.4)
```

The shader applies `sqrt()` (approximate gamma decode) then passes the result through `to_linear_rgb()` which applies the sRGB transfer function **again**. These both do gamma correction in the same direction — the image is linearized twice, making it appear too dark and overly contrasty.

**Fix:** Apply only one. Either:
- `sqrt()` as a simple gamma 2.0 output, **or**
- The proper sRGB OETF (inverse of `to_linear_rgb`), but not both.

---

### 8. Nested Dielectrics Not Handled

**File:** `src/shaders/pathtracer.wgsl:177-183`

```wgsl
var eta_t = ob_params.material.refractive_index;
if !hitParams.front_face {
    eta_t = eta_t / ray.refractive_index;
}
```

When exiting a dielectric, the code divides the material's RI by the ray's current RI. This only works for a single layer of glass. For nested dielectrics (glass inside glass, glass touching water), the RI tracking is incorrect because there's no stack tracking which medium the ray is entering.

> **PBR Book reference:** Ch. 9.5 — correct handling requires maintaining a stack of medium interfaces or using a priority-based scheme to determine the IOR ratio at each boundary.

---

## Minor Issues

### 9. Triangle Intersection: No Degenerate Triangle Guard

**File:** `src/shaders/intersects.wgsl:45-47`

```wgsl
let det = dot(e1, dirCrossE2);
let f = 1.0 / det;
```

When `det ≈ 0` (degenerate triangle or ray parallel to triangle plane), `f` becomes infinity. The subsequent multiplications can produce NaN values that may propagate through the path tracer.

> **PBR Book reference:** Ch. 6.5 — the Moller-Trumbore implementation explicitly checks `if |det| < ε` and skips the intersection.

---

### 10. No Top-Level Acceleration Structure

**File:** `src/shaders/intersects.wgsl:171-203`

The `intersect()` function iterates through **all objects** linearly. Each mesh has its own BVH, but the top-level is O(n_objects). For scenes with many objects, this becomes the bottleneck.

> **PBR Book reference:** Ch. 7 — a two-level acceleration structure (TLAS over BLAS) is standard practice.

---

### 11. Emissive Check Only Tests Red Channel

**File:** `src/shaders/pathtracer.wgsl:161`

```wgsl
if ob_params.material.emissiveness.x > 0.0 {
```

Only checks the red channel. A light with emissiveness `(0, 5, 5)` (cyan) would not be detected as emissive.

**Fix:** `if any(ob_params.material.emissiveness.xyz > vec3(0.0)) {`

---

### 12. Unnecessary `sqrt` in Distance Squared Computation

**File:** `src/shaders/pathtracer.wgsl:96`

```wgsl
let distance_squared = pow(length(direction), 2.0);
```

This computes `sqrt(x² + y² + z²)` then squares it — a wasted `sqrt`. Use `dot(direction, direction)` instead.

---

## Summary

| # | Severity | Issue | Impact |
|---|----------|-------|--------|
| 1 | **Critical** | RNG never advances | Correlated samples, structured noise, slow/incorrect convergence |
| 2 | **Critical** | SAH `b0 = b1.merge()` bug | Suboptimal BVH splits, slower rendering |
| 3 | **Critical** | Sphere intersection missing distance check | Wrong z-ordering for rays inside spheres |
| 4 | Moderate | No Russian roulette | Biased energy loss at max bounce |
| 5 | Moderate | Cube light: volume sampling + sphere PDF | Biased lighting for cube lights |
| 6 | Moderate | Rough specular missing PDF | Incorrect energy for glossy materials |
| 7 | Moderate | Double gamma correction | Image too dark |
| 8 | Moderate | Nested dielectrics broken | Wrong refraction for stacked glass |
| 9 | Minor | No degenerate triangle guard | Potential NaN propagation |
| 10 | Minor | No TLAS | O(n) top-level traversal |
| 11 | Minor | Emissive check only on red channel | Colored lights missed |
| 12 | Minor | `pow(length(), 2)` vs `dot()` | Wasted GPU cycles |
