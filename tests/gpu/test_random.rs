use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

use super::{
    assert_approx_eq, create_shader_module, require_gpu, run_test_shader, ShaderCompiler,
    GPU_TOLERANCE,
};

// ---------------------------------------------------------------------------
// CPU reference implementations -- must match the WGSL exactly
// ---------------------------------------------------------------------------

/// CPU reference: f32_zero_to_one
/// WGSL: bitcast<f32>(0x3f800000u | (seed & 0x007fffffu)) - 1.0
fn cpu_f32_zero_to_one(seed: u32) -> f32 {
    f32::from_bits(0x3f800000u32 | (seed & 0x007fffff)) - 1.0
}

/// CPU reference: f32_negone_to_one
/// WGSL: bitcast<f32>(0x40000000u | (seed & 0x007fffffu)) - 3.0
fn cpu_f32_negone_to_one(seed: u32) -> f32 {
    f32::from_bits(0x40000000u32 | (seed & 0x007fffff)) - 3.0
}

/// CPU reference: init_pcg3d -- matches the WGSL exactly
fn cpu_init_pcg3d(v: [u32; 3]) -> [u32; 3] {
    let mut s = [
        v[0].wrapping_mul(1664525).wrapping_add(1013904223),
        v[1].wrapping_mul(1664525).wrapping_add(1013904223),
        v[2].wrapping_mul(1664525).wrapping_add(1013904223),
    ];

    s[0] = s[0].wrapping_add(s[1].wrapping_mul(s[2]));
    s[1] = s[1].wrapping_add(s[2].wrapping_mul(s[0]));
    s[2] = s[2].wrapping_add(s[0].wrapping_mul(s[1]));

    let shifted = [s[0] >> 16, s[1] >> 16, s[2] >> 16];
    s[0] ^= shifted[0];
    s[1] ^= shifted[1];
    s[2] ^= shifted[2];

    s[0] = s[0].wrapping_add(s[1].wrapping_mul(s[2]));
    s[1] = s[1].wrapping_add(s[2].wrapping_mul(s[0]));
    s[2] = s[2].wrapping_add(s[0].wrapping_mul(s[1]));

    s
}

/// CPU reference: next_pcg3d
fn cpu_next_pcg3d(state: &mut [u32; 3]) {
    state[0] = state[0].wrapping_mul(1664525).wrapping_add(1013904223);
    state[1] = state[1].wrapping_mul(1664525).wrapping_add(1013904223);
    state[2] = state[2].wrapping_mul(1664525).wrapping_add(1013904223);

    state[0] = state[0].wrapping_add(state[1].wrapping_mul(state[2]));
    state[1] = state[1].wrapping_add(state[2].wrapping_mul(state[0]));
    state[2] = state[2].wrapping_add(state[0].wrapping_mul(state[1]));

    state[0] ^= state[0] >> 16;
    state[1] ^= state[1] >> 16;
    state[2] ^= state[2] >> 16;

    state[0] = state[0].wrapping_add(state[1].wrapping_mul(state[2]));
    state[1] = state[1].wrapping_add(state[2].wrapping_mul(state[0]));
    state[2] = state[2].wrapping_add(state[0].wrapping_mul(state[1]));
}

/// CPU reference: random_cosine_direction (pbrt 13.6.3)
fn cpu_random_cosine_direction(state: &mut [u32; 3]) -> [f32; 3] {
    cpu_next_pcg3d(state);
    let r1 = cpu_f32_zero_to_one(state[0]);
    let r2 = cpu_f32_zero_to_one(state[1]);
    let z = (1.0 - r2).sqrt();
    let phi = 2.0 * PI * r1;
    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    [x, y, z]
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4u([u32; 4]);

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

fn compile_random_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_random.wgsl"),
        "shaders/tests/test_random.wgsl",
    )
}

// --- f32_zero_to_one ---

#[test]
fn f32_zero_to_one_range() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..64)
        .map(|i| Vec4u([i * 7919 + 1, 0, 0, 0]))
        .collect();

    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_f32_zero_to_one", &inputs, 64);

    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.0[0] >= 0.0 && out.0[0] < 1.0,
            "seed {}: f32_zero_to_one returned {} (expected [0, 1))",
            inputs[i].0[0],
            out.0[0]
        );
    }
}

// --- f32_negone_to_one ---

#[test]
fn f32_negone_to_one_range() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..64)
        .map(|i| Vec4u([i * 4201 + 3, 0, 0, 0]))
        .collect();

    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_f32_negone_to_one", &inputs, 64);

    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.0[0] >= -1.0 && out.0[0] < 1.0,
            "seed {}: f32_negone_to_one returned {} (expected [-1, 1))",
            inputs[i].0[0],
            out.0[0]
        );
    }
}

// --- rescale ---

#[test]
fn rescale_identity() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    // Pack floats as bitcast u32
    let inputs = [Vec4u([
        0.5_f32.to_bits(),
        0.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0,
    ])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_rescale", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.5, GPU_TOLERANCE);
}

#[test]
fn rescale_scale_offset() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs = [
        Vec4u([
            0.5_f32.to_bits(),
            2.0_f32.to_bits(),
            10.0_f32.to_bits(),
            0,
        ]),
        Vec4u([
            0.0_f32.to_bits(),
            5.0_f32.to_bits(),
            10.0_f32.to_bits(),
            0,
        ]),
        Vec4u([
            1.0_f32.to_bits(),
            5.0_f32.to_bits(),
            10.0_f32.to_bits(),
            0,
        ]),
    ];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_rescale", &inputs, 3);
    assert_approx_eq(outputs[0].0[0], 6.0, GPU_TOLERANCE);
    assert_approx_eq(outputs[1].0[0], 5.0, GPU_TOLERANCE);
    // rescale(1.0, 5.0, 10.0) = (1.0 * (10-5)) + 5 = 10.0
    assert_approx_eq(outputs[2].0[0], 10.0, GPU_TOLERANCE);
}

// --- PCG3D determinism ---

#[test]
fn pcg3d_determinism() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    // Run same seed twice
    let inputs = [
        Vec4u([42, 137, 999, 0]),
        Vec4u([42, 137, 999, 0]),
    ];
    // Each invocation writes 2 output vec4s, so 4 total
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_pcg3d_determinism", &inputs, 4);

    // First invocation: outputs[0] = first state, outputs[1] = second state
    // Second invocation: outputs[2] = first state, outputs[3] = second state
    let a_first: [u32; 4] = bytemuck::cast(outputs[0]);
    let a_second: [u32; 4] = bytemuck::cast(outputs[1]);
    let b_first: [u32; 4] = bytemuck::cast(outputs[2]);
    let b_second: [u32; 4] = bytemuck::cast(outputs[3]);

    // Same seed -> same output
    assert_eq!(a_first, b_first, "same seed should produce same first state");
    assert_eq!(a_second, b_second, "same seed should produce same second state");

    // State advances
    assert_ne!(a_first, a_second, "PCG3D state should advance after next_pcg3d()");
}

// --- random_cosine_direction ---

#[test]
fn random_cosine_direction_properties() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..256)
        .map(|i| Vec4u([i * 3 + 1, i * 7 + 2, i * 13 + 3, 0]))
        .collect();

    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_cosine_direction", &inputs, 256);

    for (i, v) in outputs.iter().enumerate() {
        let len = (v.0[0] * v.0[0] + v.0[1] * v.0[1] + v.0[2] * v.0[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "sample {i}: length {len}, expected ~1.0"
        );
        assert!(
            v.0[2] >= -GPU_TOLERANCE,
            "sample {i}: z={}, expected >= 0 (cosine-weighted hemisphere)",
            v.0[2]
        );
    }
}

// --- random_uniform_direction ---

#[test]
fn random_uniform_direction_unit_length() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..256)
        .map(|i| Vec4u([i * 11 + 5, i * 17 + 7, i * 23 + 11, 0]))
        .collect();

    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_uniform_direction", &inputs, 256);

    for (i, v) in outputs.iter().enumerate() {
        let len = (v.0[0] * v.0[0] + v.0[1] * v.0[1] + v.0[2] * v.0[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "sample {i}: length {len}, expected ~1.0"
        );
    }
}

// --- random_in_unit_sphere ---

#[test]
fn random_in_unit_sphere_inside() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..256)
        .map(|i| Vec4u([i * 31 + 1, i * 37 + 2, i * 41 + 3, 0]))
        .collect();

    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_in_unit_sphere", &inputs, 256);

    for (i, v) in outputs.iter().enumerate() {
        let len = (v.0[0] * v.0[0] + v.0[1] * v.0[1] + v.0[2] * v.0[2]).sqrt();
        assert!(
            len <= 1.0 + GPU_TOLERANCE,
            "sample {i}: length {len}, expected <= 1.0"
        );
    }
}

// ===========================================================================
// CORRECTNESS TESTS: verify GPU output matches CPU reference bit-for-bit
// ===========================================================================

// --- f32_zero_to_one: exact match against CPU reference ---

#[test]
fn f32_zero_to_one_exact_values() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let test_seeds: Vec<u32> = vec![
        0, 1, 0x007fffff, 0xFFFFFFFF, 0x00400000, 0xDEADBEEF, 42, 1000000,
        0x3f800000, // would produce 1.0 in the float — make sure masking prevents it
    ];
    let inputs: Vec<Vec4u> = test_seeds.iter().map(|&s| Vec4u([s, 0, 0, 0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_f32_zero_to_one", &inputs, n);

    for (i, &seed) in test_seeds.iter().enumerate() {
        let expected = cpu_f32_zero_to_one(seed);
        assert!(
            (outputs[i].0[0] - expected).abs() < 1e-7,
            "f32_zero_to_one(0x{:08x}): GPU={}, CPU={}",
            seed, outputs[i].0[0], expected
        );
    }
}

// --- f32_negone_to_one: exact match against CPU reference ---

#[test]
fn f32_negone_to_one_exact_values() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let test_seeds: Vec<u32> = vec![
        0, 1, 0x007fffff, 0xFFFFFFFF, 0x00400000, 0xDEADBEEF, 42, 1000000,
    ];
    let inputs: Vec<Vec4u> = test_seeds.iter().map(|&s| Vec4u([s, 0, 0, 0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_f32_negone_to_one", &inputs, n);

    for (i, &seed) in test_seeds.iter().enumerate() {
        let expected = cpu_f32_negone_to_one(seed);
        assert!(
            (outputs[i].0[0] - expected).abs() < 1e-7,
            "f32_negone_to_one(0x{:08x}): GPU={}, CPU={}",
            seed, outputs[i].0[0], expected
        );
    }
}

// --- PCG3D: verify GPU state matches CPU reference bit-exact ---

#[test]
fn pcg3d_matches_cpu_reference() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let test_seeds: Vec<[u32; 3]> = vec![
        [0, 0, 0],
        [1, 2, 3],
        [42, 137, 999],
        [0xDEADBEEF, 0xCAFEBABE, 0x12345678],
        [u32::MAX, u32::MAX, u32::MAX],
    ];

    let inputs: Vec<Vec4u> = test_seeds.iter().map(|s| Vec4u([s[0], s[1], s[2], 0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_pcg3d_determinism", &inputs, n * 2);

    for (i, seed) in test_seeds.iter().enumerate() {
        // CPU reference
        let mut cpu_state = cpu_init_pcg3d(*seed);
        let cpu_first = cpu_state;
        cpu_next_pcg3d(&mut cpu_state);
        let cpu_second = cpu_state;

        // GPU output (2 vec4s per invocation, bitcast back to u32)
        let gpu_first: [u32; 4] = bytemuck::cast(outputs[i * 2]);
        let gpu_second: [u32; 4] = bytemuck::cast(outputs[i * 2 + 1]);

        assert_eq!(
            [gpu_first[0], gpu_first[1], gpu_first[2]],
            cpu_first,
            "PCG3D init({:?}): GPU first state {:?} != CPU {:?}",
            seed, &gpu_first[..3], cpu_first
        );
        assert_eq!(
            [gpu_second[0], gpu_second[1], gpu_second[2]],
            cpu_second,
            "PCG3D init({:?})+next: GPU second state {:?} != CPU {:?}",
            seed, &gpu_second[..3], cpu_second
        );
    }
}

// --- Cosine-weighted hemisphere: verify against CPU reference ---

#[test]
fn random_cosine_direction_matches_cpu() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let test_seeds: Vec<[u32; 3]> = vec![
        [1, 2, 3],
        [100, 200, 300],
        [42, 137, 999],
        [7, 13, 19],
    ];

    let inputs: Vec<Vec4u> = test_seeds.iter().map(|s| Vec4u([s[0], s[1], s[2], 0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_random_cosine_direction", &inputs, n);

    for (i, seed) in test_seeds.iter().enumerate() {
        let mut cpu_state = cpu_init_pcg3d(*seed);
        let expected = cpu_random_cosine_direction(&mut cpu_state);
        let gpu = [outputs[i].0[0], outputs[i].0[1], outputs[i].0[2]];
        for c in 0..3 {
            assert!(
                (gpu[c] - expected[c]).abs() < 1e-4,
                "cosine_direction seed {:?} component {}: GPU={}, CPU={}",
                seed, c, gpu[c], expected[c]
            );
        }
    }
}

// --- Cosine sampling: statistical test (pbrt 13.6.3) ---
// E[cos(theta)] for cosine-weighted hemisphere = 2/3
// E[cos²(theta)] = 1/2
// These catch wrong PDF implementations even when samples are "in range"

#[test]
fn cosine_sampling_mean_cos_theta() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let n = 4096u32;
    let inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 97 + 13, i * 53 + 7, i * 31 + 41, 0]))
        .collect();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_cosine_direction", &inputs, n as usize);

    // z component = cos(theta) for local hemisphere
    let mean_cos: f64 = outputs.iter().map(|v| v.0[2] as f64).sum::<f64>() / n as f64;
    // E[cos(theta)] = integral(cos^2(theta)*sin(theta), 0..pi/2) / integral(cos(theta)*sin(theta), 0..pi/2)
    //               = (1/3) / (1/2) = 2/3
    assert!(
        (mean_cos - 2.0 / 3.0).abs() < 0.03,
        "E[cos(theta)] = {mean_cos}, expected ~0.6667 (pbrt cosine-weighted hemisphere)"
    );

    // E[cos²(theta)] = 1/2 for cosine-weighted sampling
    let mean_cos2: f64 = outputs.iter().map(|v| (v.0[2] as f64).powi(2)).sum::<f64>() / n as f64;
    assert!(
        (mean_cos2 - 0.5).abs() < 0.03,
        "E[cos²(theta)] = {mean_cos2}, expected ~0.5"
    );
}

// --- Uniform sphere sampling: statistical test ---
// E[z] should be 0 (symmetric about equator), E[z²] = 1/3

#[test]
fn uniform_direction_mean_z() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let n = 4096u32;
    let inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 101 + 3, i * 59 + 11, i * 37 + 23, 0]))
        .collect();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_uniform_direction", &inputs, n as usize);

    let mean_z: f64 = outputs.iter().map(|v| v.0[2] as f64).sum::<f64>() / n as f64;
    assert!(
        mean_z.abs() < 0.05,
        "E[z] for uniform sphere = {mean_z}, expected ~0.0"
    );

    let mean_z2: f64 = outputs.iter().map(|v| (v.0[2] as f64).powi(2)).sum::<f64>() / n as f64;
    assert!(
        (mean_z2 - 1.0 / 3.0).abs() < 0.03,
        "E[z²] for uniform sphere = {mean_z2}, expected ~0.333"
    );
}

// --- Uniform hemisphere: statistical test ---
// E[z] for uniform hemisphere = 1/2, E[z²] = 1/3

#[test]
fn uniform_hemisphere_mean_z() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let n = 4096u32;
    let inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 89 + 5, i * 67 + 13, i * 43 + 29, 0]))
        .collect();
    let outputs: Vec<Vec4f> = run_test_shader(
        &ctx,
        &module,
        "test_random_uniform_on_hemisphere",
        &inputs,
        n as usize,
    );

    let mean_z: f64 = outputs.iter().map(|v| v.0[2] as f64).sum::<f64>() / n as f64;
    assert!(
        (mean_z - 0.5).abs() < 0.03,
        "E[z] for uniform hemisphere = {mean_z}, expected ~0.5"
    );
}

// --- random_in_unit_sphere: statistical test ---
// E[r²] for uniform volume sampling = 3/5 (mean squared radius)

#[test]
fn unit_sphere_mean_squared_radius() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let n = 4096u32;
    let inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 73 + 1, i * 79 + 2, i * 83 + 3, 0]))
        .collect();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_random_in_unit_sphere", &inputs, n as usize);

    let mean_r2: f64 = outputs
        .iter()
        .map(|v| {
            let r2 = v.0[0] as f64 * v.0[0] as f64
                + v.0[1] as f64 * v.0[1] as f64
                + v.0[2] as f64 * v.0[2] as f64;
            r2
        })
        .sum::<f64>()
        / n as f64;
    // E[r²] = integral(r^4 * 3, 0..1) = 3/5 = 0.6
    assert!(
        (mean_r2 - 0.6).abs() < 0.04,
        "E[r²] for unit sphere = {mean_r2}, expected ~0.6"
    );
}

// --- random_uniform_on_hemisphere ---

#[test]
fn random_uniform_on_hemisphere_properties() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_random_module(&ctx, &mut compiler);

    let inputs: Vec<Vec4u> = (0..256)
        .map(|i| Vec4u([i * 43 + 1, i * 47 + 2, i * 53 + 3, 0]))
        .collect();

    let outputs: Vec<Vec4f> = run_test_shader(
        &ctx,
        &module,
        "test_random_uniform_on_hemisphere",
        &inputs,
        256,
    );

    for (i, v) in outputs.iter().enumerate() {
        let len = (v.0[0] * v.0[0] + v.0[1] * v.0[1] + v.0[2] * v.0[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "sample {i}: length {len}, expected ~1.0"
        );
        assert!(
            v.0[2] >= -GPU_TOLERANCE,
            "sample {i}: z={}, expected >= 0 (hemisphere)",
            v.0[2]
        );
    }
}
