#import rengin::constants::PI
#import rengin::random::{
    init_pcg3d, next_pcg3d, rand_pcg3d,
    f32_zero_to_one,
    random_cosine_direction, random_uniform_on_hemisphere,
}
#import rengin::pathtracer_utils::onb
#import rengin::pathtracer_math::{
    scattering_pdf, mis_pdf, snell_refract,
    specular_weight, path_regularization_roughness,
    regularize_indirect_radiance,
    determine_eta_t, cube_face_select,
}

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// ============================================================================
// White furnace test — no MIS (calls production scattering_pdf)
// ============================================================================
@compute @workgroup_size(1)
fn test_white_furnace_no_mis(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = inputs_4f[id.x];
    init_pcg3d(vec3<u32>(bitcast<u32>(seed.x), bitcast<u32>(seed.y), bitcast<u32>(seed.z)));

    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let basis = onb(normal);
    let colour = vec3<f32>(1.0, 1.0, 1.0);

    let cosine_dir = basis * random_cosine_direction();
    let direction = normalize(cosine_dir);

    let s_pdf = scattering_pdf(direction, normal);
    let pdf = s_pdf; // no MIS

    var throughput_factor = vec3<f32>(0.0);
    if pdf > 0.0 {
        throughput_factor = colour * s_pdf / pdf;
    }

    let cos_theta = dot(direction, normal);
    outputs_4f[id.x] = vec4<f32>(throughput_factor, cos_theta);
}

// ============================================================================
// White furnace test — with MIS (calls production scattering_pdf + mis_pdf)
// ============================================================================
@compute @workgroup_size(1)
fn test_white_furnace_mis(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = inputs_4f[id.x];
    init_pcg3d(vec3<u32>(bitcast<u32>(seed.x), bitcast<u32>(seed.y), bitcast<u32>(seed.z)));

    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let basis = onb(normal);
    let colour = vec3<f32>(1.0, 1.0, 1.0);
    let p_scatter = 0.5;

    next_pcg3d();
    var direction: vec3<f32>;
    if f32_zero_to_one(rand_pcg3d.y) < p_scatter {
        direction = normalize(basis * random_cosine_direction());
    } else {
        direction = normalize(basis * random_uniform_on_hemisphere());
    }

    let s_pdf = scattering_pdf(direction, normal);
    let v_light_pdf = 1.0 / (2.0 * PI); // uniform hemisphere
    let pdf = mis_pdf(s_pdf, v_light_pdf, p_scatter);

    var throughput_factor = vec3<f32>(0.0);
    if pdf > 0.0 {
        throughput_factor = colour * s_pdf / pdf;
    }

    let cos_theta = dot(direction, normal);
    outputs_4f[id.x] = vec4<f32>(throughput_factor, cos_theta);
}

// ============================================================================
// Specular weight (calls production specular_weight)
// ============================================================================
@compute @workgroup_size(1)
fn test_specular_weight(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 2u;
    let params = inputs_4f[base];
    let colour_in = inputs_4f[base + 1u];
    let reflective = params.x;
    let colour = colour_in.xyz;
    let w = specular_weight(colour, reflective);
    outputs_4f[id.x] = vec4<f32>(w, 0.0);
}

// ============================================================================
// Emissive hit: throughput * emissiveness (trivial, stays inline)
// ============================================================================
@compute @workgroup_size(1)
fn test_emissive_hit(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 2u;
    let throughput = inputs_4f[base].xyz;
    let emissiveness = inputs_4f[base + 1u].xyz;
    outputs_4f[id.x] = vec4<f32>(throughput * emissiveness, 0.0);
}

// ============================================================================
// Path regularization (calls production path_regularization_roughness)
// ============================================================================
@compute @workgroup_size(1)
fn test_path_regularization(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let roughness = path_regularization_roughness(inp.x, inp.y);
    outputs_4f[id.x] = vec4<f32>(roughness, 0.0, 0.0, 0.0);
}

// ============================================================================
// Progressive averaging (trivial mix, stays inline)
// ============================================================================
@compute @workgroup_size(1)
fn test_progressive_average(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 3u;
    let old_color = inputs_4f[base].xyz;
    let new_color = inputs_4f[base + 1u].xyz;
    let n = inputs_4f[base + 2u].x;
    let scale = 1.0 / (n + 1.0);
    let result = mix(old_color, new_color, vec3<f32>(scale));
    outputs_4f[id.x] = vec4<f32>(result, 0.0);
}

// ============================================================================
// Sample sanitization: negative numerical noise is removed, HDR is preserved.
// ============================================================================
@compute @workgroup_size(1)
fn test_sample_radiance(@builtin(global_invocation_id) id: vec3<u32>) {
    let color = inputs_4f[id.x];
    outputs_4f[id.x] = max(color, vec4<f32>(0.0));
}

// ============================================================================
// Indirect firefly regularization: rgb=input, w=has_diffuse; fixed limit=3.
// ============================================================================
@compute @workgroup_size(1)
fn test_regularize_indirect_radiance(@builtin(global_invocation_id) id: vec3<u32>) {
    let color = inputs_4f[id.x];
    let result = regularize_indirect_radiance(color.xyz, 3.0, color.w > 0.5);
    outputs_4f[id.x] = vec4<f32>(result, 0.0);
}

// ============================================================================
// RI tracking (calls production determine_eta_t + snell_refract)
// ============================================================================
@compute @workgroup_size(1)
fn test_ri_tracking(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let ray_ri = inp.x;
    let material_ri = inp.y;
    let front_face = inp.z > 0.5;
    let cos_i = inp.w;

    // determine_eta_t: production function for RI state machine
    let eta_t = determine_eta_t(ray_ri, material_ri, front_face);

    // snell_refract: production function, returns .w < 0 for TIR
    let n_ratio = ray_ri / eta_t;
    let refraction = snell_refract(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), cos_i, n_ratio);

    // next_ri = eta_t if can refract, else stays ray_ri (same logic as pipeline)
    var next_ri = ray_ri;
    if refraction.w > 0.0 {
        next_ri = eta_t;
    }

    outputs_4f[id.x] = vec4<f32>(next_ri, eta_t, 0.0, 0.0);
}

// ============================================================================
// Cube face sampling (calls production cube_face_select)
// ============================================================================
@group(0) @binding(2) var<storage, read> inputs_4u: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> outputs_4f_b: array<vec4<f32>>;

@compute @workgroup_size(1)
fn test_cube_face_sampling(@builtin(global_invocation_id) id: vec3<u32>) {
    let params = inputs_4f[id.x];
    let seed = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(seed.x, seed.y, seed.z));

    let scale = vec3<f32>(params.x, params.y, params.z);
    next_pcg3d();
    let u = f32_zero_to_one(rand_pcg3d.x);
    let face = cube_face_select(scale, u);

    outputs_4f_b[id.x] = vec4<f32>(f32(face), 0.0, 0.0, 0.0);
}

// ============================================================================
// MIS throughput update (calls production scattering_pdf + mis_pdf)
// ============================================================================
@compute @workgroup_size(1)
fn test_mis_throughput_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = inputs_4u[id.x];
    let params = inputs_4f[id.x];
    init_pcg3d(vec3<u32>(seed.x, seed.y, seed.z));

    let colour = params.xyz;
    let simulated_light_pdf = params.w;

    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let basis = onb(normal);
    let p_scatter = 0.5;

    next_pcg3d();
    var direction: vec3<f32>;
    if f32_zero_to_one(rand_pcg3d.y) < p_scatter {
        direction = normalize(basis * random_cosine_direction());
    } else {
        direction = normalize(basis * random_uniform_on_hemisphere());
    }

    let s_pdf = scattering_pdf(direction, normal);
    let pdf = mis_pdf(s_pdf, simulated_light_pdf, p_scatter);

    var throughput_factor = vec3<f32>(0.0);
    if pdf > 0.0 {
        throughput_factor = colour * s_pdf / pdf;
    }

    outputs_4f_b[id.x] = vec4<f32>(throughput_factor, pdf);
}
