#import rengin::random::{init_pcg3d, next_pcg3d, rand_pcg3d, f32_zero_to_one}
#import rengin::fresnel::schlick_lazanyi
#import rengin::pathtracer_math::{
    snell_refract, russian_roulette_q, russian_roulette_adjust,
}

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// ---- Snell's law refraction direction (calls production snell_refract) ----
@compute @workgroup_size(1)
fn test_snell_refraction(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 3u;
    let inp0 = inputs_4f[base];
    let inp1 = inputs_4f[base + 1u];
    let inp2 = inputs_4f[base + 2u];

    let eyev = inp0.xyz;
    let cos_i = inp0.w;
    let normalv = inp1.xyz;
    let ri = inp2.x;
    let eta_t = inp2.y;
    let n_ratio = ri / eta_t;

    let result = snell_refract(eyev, normalv, cos_i, n_ratio);
    outputs_4f[id.x] = result;
}

// ---- Russian roulette (calls production functions) ----
@compute @workgroup_size(1)
fn test_russian_roulette(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let throughput = vec3<f32>(inp.x, inp.y, inp.z);

    let q = russian_roulette_q(throughput);
    if q < 0.0 {
        // No RR applied
        outputs_4f[id.x] = vec4<f32>(-1.0, throughput.x, throughput.y, throughput.z);
    } else {
        let adjusted = russian_roulette_adjust(throughput, q);
        outputs_4f[id.x] = vec4<f32>(q, adjusted.x, adjusted.y, adjusted.z);
    }
}

// ---- Russian roulette statistical test ----
@group(0) @binding(2) var<storage, read> inputs_4u: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> outputs_4f_b: array<vec4<f32>>;

@compute @workgroup_size(1)
fn test_russian_roulette_statistical(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = inputs_4u[id.x];
    let params = inputs_4f[id.x];
    let throughput = params.xyz;

    init_pcg3d(vec3<u32>(seed.x, seed.y, seed.z));

    let q = russian_roulette_q(throughput);
    if q < 0.0 {
        outputs_4f_b[id.x] = vec4<f32>(throughput, 1.0);
    } else {
        next_pcg3d();
        if f32_zero_to_one(rand_pcg3d.z) < q {
            outputs_4f_b[id.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        } else {
            let adjusted = russian_roulette_adjust(throughput, q);
            outputs_4f_b[id.x] = vec4<f32>(adjusted, 1.0);
        }
    }
}

// ---- Total internal reflection check ----
// Uses the same production functions as the pipeline: snell_refract for TIR detection,
// schlick_lazanyi for reflectance.
@compute @workgroup_size(1)
fn test_tir_check(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let ri = inp.x;
    let eta_t = inp.y;
    let cos_i = inp.z;

    let n_ratio = ri / eta_t;
    let refraction = snell_refract(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), cos_i, n_ratio);
    let reflectance = schlick_lazanyi(cos_i, eta_t, 0.0);

    // output: (is_tir, reflectance, 0, 0)
    // refraction.w < 0 means TIR (same check as pipeline)
    outputs_4f[id.x] = vec4<f32>(select(0.0, 1.0, refraction.w < 0.0), reflectance, 0.0, 0.0);
}
