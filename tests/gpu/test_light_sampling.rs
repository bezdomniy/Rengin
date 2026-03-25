use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

use super::{
    assert_approx_eq, create_shader_module, dispatch_and_readback, require_gpu, run_test_shader,
    BufferSpec, ShaderCompiler, GPU_TOLERANCE,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4u([u32; 4]);

fn compile_light_sampling_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_light_sampling.wgsl"),
        "shaders/tests/test_light_sampling.wgsl",
    )
}

// ---------------------------------------------------------------------------
// CPU reference: light PDF for sphere (pbrt 12.2.1)
// ---------------------------------------------------------------------------

fn cpu_light_pdf_sphere(radius: f32, distance: f32) -> f32 {
    let cos_theta_max = (1.0 - radius * radius / (distance * distance)).sqrt();
    let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
    1.0 / solid_angle
}

// ===========================================================================
// light_pdf tests (pbrt 12.2.1: solid angle subtended by sphere)
// ===========================================================================

#[test]
fn light_pdf_sphere_unit_distance_5() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Unit sphere at distance 5
    let inputs = [Vec4f([1.0, 5.0, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_light_pdf_sphere", &inputs, 1);
    let expected = cpu_light_pdf_sphere(1.0, 5.0);
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
}

#[test]
fn light_pdf_sphere_radius_2_distance_10() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    let inputs = [Vec4f([2.0, 10.0, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_light_pdf_sphere", &inputs, 1);
    let expected = cpu_light_pdf_sphere(2.0, 10.0);
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
}

#[test]
fn light_pdf_sphere_close() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Radius 1 at distance 1.5 — large solid angle
    let inputs = [Vec4f([1.0, 1.5, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_light_pdf_sphere", &inputs, 1);
    let expected = cpu_light_pdf_sphere(1.0, 1.5);
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
    // PDF should be positive
    assert!(outputs[0].0[0] > 0.0, "PDF must be positive");
}

#[test]
fn light_pdf_sphere_far_away() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Radius 1 at distance 100 — very small solid angle, high PDF
    let inputs = [Vec4f([1.0, 100.0, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_light_pdf_sphere", &inputs, 1);
    let expected = cpu_light_pdf_sphere(1.0, 100.0);
    // Large PDF values need relative tolerance due to GPU float precision
    let rel_err = (outputs[0].0[0] - expected).abs() / expected;
    assert!(rel_err < 0.005, "light_pdf far away: GPU={}, CPU={}, rel_err={}", outputs[0].0[0], expected, rel_err);
}

#[test]
fn light_pdf_sphere_multiple_distances() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Sweep distances: PDF should decrease as solid angle increases (closer = smaller PDF)
    let distances = [2.0_f32, 3.0, 5.0, 10.0, 20.0, 50.0];
    let inputs: Vec<Vec4f> = distances.iter().map(|&d| Vec4f([1.0, d, 0.0, 0.0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_light_pdf_sphere", &inputs, n);

    for (i, &d) in distances.iter().enumerate() {
        let expected = cpu_light_pdf_sphere(1.0, d);
        // Use relative tolerance for varying magnitudes
        let rel_err = (outputs[i].0[0] - expected).abs() / expected.max(1e-6);
        assert!(
            rel_err < 0.001,
            "d={d}: GPU={}, CPU={}, rel_err={}",
            outputs[i].0[0], expected, rel_err
        );
    }

    // PDF should increase with distance (smaller solid angle → higher PDF)
    for i in 1..n {
        assert!(
            outputs[i].0[0] > outputs[i - 1].0[0],
            "PDF should increase with distance: d={} pdf={} vs d={} pdf={}",
            distances[i], outputs[i].0[0], distances[i - 1], outputs[i - 1].0[0]
        );
    }
}

// ===========================================================================
// scattering_pdf tests (pbrt 13.6.3: cosine-weighted hemisphere PDF)
// ===========================================================================

#[test]
fn scattering_pdf_normal_incidence() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // direction = (0,0,1), normal = (0,0,1): cos(0) = 1, PDF = 1/PI
    let inputs = [Vec4f([0.0, 0.0, 1.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_scattering_pdf", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 1.0 / PI, GPU_TOLERANCE);
}

#[test]
fn scattering_pdf_45_degrees() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // direction at 45° from normal (0,0,1)
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let inputs = [Vec4f([inv_sqrt2, 0.0, inv_sqrt2, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_scattering_pdf", &inputs, 1);
    let expected = inv_sqrt2 / PI;
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
}

#[test]
fn scattering_pdf_grazing() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // direction at ~89° from normal
    let angle = 89.0_f32.to_radians();
    let dir = [angle.sin(), 0.0, angle.cos()];
    let inputs = [Vec4f([dir[0], dir[1], dir[2], 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_scattering_pdf", &inputs, 1);
    let expected = angle.cos() / PI;
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
    assert!(outputs[0].0[0] > 0.0, "grazing PDF should still be positive");
}

#[test]
fn scattering_pdf_behind_surface() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // direction opposite to normal: cos < 0, PDF = max(0, ...) = 0
    let inputs = [Vec4f([0.0, 0.0, -1.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_scattering_pdf", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.0, GPU_TOLERANCE);
}

#[test]
fn scattering_pdf_exactly_perpendicular() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // direction perpendicular to normal: cos(90°) = 0, PDF = 0
    let inputs = [Vec4f([1.0, 0.0, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_scattering_pdf", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.0, GPU_TOLERANCE);
}

// ===========================================================================
// MIS weight tests (pbrt 13.10.1: balance heuristic)
// ===========================================================================

#[test]
fn mis_weight_equal_pdfs() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // scattering_pdf = light_pdf = 0.5, p_scatter = 0.5
    // result = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
    let inputs = [Vec4f([0.5, 0.5, 0.5, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_mis_weight", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.5, GPU_TOLERANCE);
}

#[test]
fn mis_weight_light_dominant() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // scattering_pdf = 0.01, light_pdf = 10.0, p_scatter = 0.5
    // result = 0.5 * 0.01 + 0.5 * 10.0 = 5.005
    let inputs = [Vec4f([0.01, 10.0, 0.5, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_mis_weight", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 5.005, GPU_TOLERANCE);
}

#[test]
fn mis_weight_scatter_dominant() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // scattering_pdf = 5.0, light_pdf = 0.001, p_scatter = 0.5
    // result = 0.5 * 5.0 + 0.5 * 0.001 = 2.5005
    let inputs = [Vec4f([5.0, 0.001, 0.5, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_mis_weight", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 2.5005, GPU_TOLERANCE);
}

#[test]
fn mis_weight_zero_light_pdf() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // When light_pdf = 0 (no light hit), only scatter term remains
    let inputs = [Vec4f([0.3, 0.0, 0.5, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_mis_weight", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.15, GPU_TOLERANCE);
}

#[test]
fn mis_weight_always_positive() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Various combinations should all produce non-negative weights
    let inputs = vec![
        Vec4f([0.1, 0.1, 0.5, 0.0]),
        Vec4f([1.0, 0.0, 0.5, 0.0]),
        Vec4f([0.0, 1.0, 0.5, 0.0]),
        Vec4f([0.3, 0.7, 0.5, 0.0]),
        Vec4f([5.0, 5.0, 0.5, 0.0]),
    ];
    let n = inputs.len();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_mis_weight", &inputs, n);

    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.0[0] >= 0.0,
            "MIS weight must be non-negative, got {} for case {}",
            out.0[0], i
        );
    }
}

// ===========================================================================
// random_to_sphere tests (pbrt 13.6.3: cone sampling)
// ===========================================================================

#[test]
fn random_to_sphere_cone_bounds() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // Unit sphere at distance_squared = 25 (distance = 5)
    // cos_theta_max = sqrt(1 - 1/25) = sqrt(0.96) ≈ 0.9798
    let radius = 1.0_f32;
    let dist_sq = 25.0_f32;
    let cos_theta_max = (1.0 - radius * radius / dist_sq).sqrt();

    let n = 4096u32;
    let float_inputs: Vec<Vec4f> = (0..n).map(|_| Vec4f([radius, dist_sq, 0.0, 0.0])).collect();
    let seed_inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 97 + 13, i * 53 + 7, i * 31 + 41, 0]))
        .collect();

    let float_bytes = bytemuck::cast_slice(&float_inputs).to_vec();
    let seed_bytes = bytemuck::cast_slice(&seed_inputs).to_vec();
    let output_bytes = vec![0u8; n as usize * std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, float_bytes),
        BufferSpec::storage_readonly(1, vec![0u8; output_bytes.len()]), // unused outputs_4f
        BufferSpec::storage_readonly(2, seed_bytes),
        BufferSpec::storage_readwrite(3, output_bytes),
    ];

    let results = dispatch_and_readback(&ctx, &module, "test_random_to_sphere", [n, 1, 1], &buffers);
    let outputs: &[Vec4f] = bytemuck::cast_slice(&results[0]);

    for (i, v) in outputs.iter().enumerate() {
        let len = (v.0[0] * v.0[0] + v.0[1] * v.0[1] + v.0[2] * v.0[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "sample {i}: length {len}, expected ~1.0"
        );
        // z component should be >= cos_theta_max (within cone)
        assert!(
            v.0[2] >= cos_theta_max - 0.01,
            "sample {i}: z={} < cos_theta_max={} (outside cone)",
            v.0[2], cos_theta_max
        );
    }
}

#[test]
fn random_to_sphere_mean_z() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_light_sampling_module(&ctx, &mut compiler);

    // For uniform cone sampling, E[z] = (1 + cos_theta_max) / 2
    let radius = 1.0_f32;
    let dist_sq = 9.0_f32; // distance = 3
    let cos_theta_max = (1.0 - radius * radius / dist_sq).sqrt();

    let n = 4096u32;
    let float_inputs: Vec<Vec4f> = (0..n).map(|_| Vec4f([radius, dist_sq, 0.0, 0.0])).collect();
    let seed_inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 89 + 5, i * 67 + 13, i * 43 + 29, 0]))
        .collect();

    let float_bytes = bytemuck::cast_slice(&float_inputs).to_vec();
    let seed_bytes = bytemuck::cast_slice(&seed_inputs).to_vec();
    let output_bytes = vec![0u8; n as usize * std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, float_bytes),
        BufferSpec::storage_readonly(1, vec![0u8; output_bytes.len()]),
        BufferSpec::storage_readonly(2, seed_bytes),
        BufferSpec::storage_readwrite(3, output_bytes),
    ];

    let results = dispatch_and_readback(&ctx, &module, "test_random_to_sphere", [n, 1, 1], &buffers);
    let outputs: &[Vec4f] = bytemuck::cast_slice(&results[0]);

    let mean_z: f64 = outputs.iter().map(|v| v.0[2] as f64).sum::<f64>() / n as f64;
    let expected_mean_z = (1.0 + cos_theta_max as f64) / 2.0;

    assert!(
        (mean_z - expected_mean_z).abs() < 0.03,
        "E[z] for cone sampling = {mean_z}, expected ~{expected_mean_z}"
    );
}
