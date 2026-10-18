use bytemuck::{Pod, Zeroable};

use super::{
    assert_approx_eq, create_shader_module, require_gpu, run_test_shader, ShaderCompiler,
    GPU_TOLERANCE,
};

// ---------------------------------------------------------------------------
// CPU reference: exact dielectric Fresnel (pbrt 9.5)
// Used to verify the Schlick approximation is reasonably accurate
// ---------------------------------------------------------------------------

/// Exact Fresnel reflectance for dielectric interface (unpolarized light).
/// pbrt 4th ed, Section 9.5.1, Eq 9.17
fn fresnel_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_theta_i.abs().min(1.0);
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = (eta_i / eta_t).powi(2) * sin2_i;

    // Total internal reflection
    if sin2_t >= 1.0 {
        return 1.0;
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    let r_parl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    (r_parl * r_parl + r_perp * r_perp) / 2.0
}

/// CPU reference for schlick_lazanyi as implemented in the shader
fn cpu_schlick_lazanyi(cos_i: f32, eta_t: f32, k: f32) -> f32 {
    ((eta_t - 1.0).powi(2) + 4.0 * eta_t * (1.0 - cos_i).powi(5) + k.powi(2))
        / ((eta_t + 1.0).powi(2) + k.powi(2))
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

fn compile_fresnel_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_fresnel.wgsl"),
        "shaders/tests/test_fresnel.wgsl",
    )
}

// --- _schlick tests ---

#[test]
fn schlick_base_normal_incidence() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // (cos_i, r0, _, _)
    let inputs = [Vec4f([1.0, 0.04, 0.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_base", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.04, GPU_TOLERANCE);
}

#[test]
fn schlick_base_grazing_angle() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.0, 0.04, 0.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_base", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 1.0, GPU_TOLERANCE);
}

// --- schlick tests ---

#[test]
fn schlick_air_glass_normal() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // (cos_i, eta_t, eta_i, _)
    let inputs = [Vec4f([1.0, 1.5, 1.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.04, GPU_TOLERANCE);
}

#[test]
fn schlick_air_glass_grazing() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.0, 1.5, 1.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 1.0, GPU_TOLERANCE);
}

#[test]
fn schlick_equal_media() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.5, 1.0, 1.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 1);
    // r0 = 0, but Schlick still returns (1-0)*(1-0.5)^5 = 0.03125
    assert_approx_eq(outputs[0].0[0], 0.03125, GPU_TOLERANCE);
}

#[test]
fn schlick_diamond_normal() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([1.0, 2.42, 1.0, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 1);
    let expected = ((1.0_f32 - 2.42) / (1.0 + 2.42)).powi(2);
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
}

// --- schlick_lazanyi tests ---

#[test]
fn schlick_lazanyi_normal_k0() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // (cos_i, eta_t, k, _)
    let inputs = [Vec4f([1.0, 1.5, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_schlick_lazanyi", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.04, GPU_TOLERANCE);
}

#[test]
fn schlick_lazanyi_grazing_k0() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.0, 1.5, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_schlick_lazanyi", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 1.0, GPU_TOLERANCE);
}

#[test]
fn schlick_lazanyi_with_absorption() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.5, 1.5, 0.5, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_schlick_lazanyi", &inputs, 1);
    let expected = 0.6875_f32 / 6.5;
    assert_approx_eq(outputs[0].0[0], expected, GPU_TOLERANCE);
}

// ===========================================================================
// COMPREHENSIVE CORRECTNESS TESTS (pbrt-derived)
// ===========================================================================

// --- Schlick vs exact Fresnel: approximation quality (pbrt 9.5.2) ---
// Schlick should be within ~1% of exact Fresnel for most angles for
// common dielectric materials. If not, the approximation is wrong.

#[test]
fn schlick_vs_exact_fresnel_glass() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // Test at 10-degree increments from normal to grazing for air->glass (1.0 -> 1.5)
    let eta_i = 1.0_f32;
    let eta_t = 1.5_f32;
    let angles: Vec<f32> = (0..9).map(|i| (i as f32 * 10.0).to_radians().cos()).collect();
    // Add near-grazing
    let mut cos_values: Vec<f32> = angles;
    cos_values.push(0.05);
    cos_values.push(0.01);

    let inputs: Vec<Vec4f> = cos_values.iter().map(|&c| Vec4f([c, eta_t, eta_i, 0.0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, n);

    for (i, &cos_i) in cos_values.iter().enumerate() {
        let exact = fresnel_dielectric(cos_i, eta_i, eta_t);
        let schlick_val = outputs[i].0[0];
        // Schlick is an approximation. Per pbrt 9.5.2, the worst case is at
        // intermediate angles near grazing. Allow 10% relative error or 0.05 absolute.
        let err = (schlick_val - exact).abs();
        let rel_err = if exact > 0.01 { err / exact } else { err };
        assert!(
            rel_err < 0.10 || err < 0.05,
            "Schlick(cos_i={cos_i:.3}, glass) = {schlick_val:.6}, exact = {exact:.6}, err = {err:.6} ({:.1}%)",
            rel_err * 100.0
        );
    }
}

#[test]
fn schlick_vs_exact_fresnel_water() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // Water: eta = 1.33
    let eta_i = 1.0_f32;
    let eta_t = 1.33_f32;
    let cos_values: Vec<f32> = (0..10).map(|i| 1.0 - i as f32 * 0.1).collect();
    let inputs: Vec<Vec4f> = cos_values.iter().map(|&c| Vec4f([c, eta_t, eta_i, 0.0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, n);

    for (i, &cos_i) in cos_values.iter().enumerate() {
        let exact = fresnel_dielectric(cos_i, eta_i, eta_t);
        let schlick_val = outputs[i].0[0];
        let err = (schlick_val - exact).abs();
        let rel_err = if exact > 0.01 { err / exact } else { err };
        assert!(
            rel_err < 0.15 || err < 0.05,
            "Schlick(cos_i={cos_i:.3}, water) = {schlick_val:.6}, exact = {exact:.6}, err = {err:.6}"
        );
    }
}

// --- Schlick symmetry: R(eta_i->eta_t) at cos_i == R(eta_t->eta_i) at cos_t ---
// (Helmholtz reciprocity, pbrt 9.5)

#[test]
fn schlick_reciprocity() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let eta_i = 1.0_f32;
    let eta_t = 1.5_f32;
    let cos_i = 0.8_f32;
    // Snell: sin_t = (eta_i/eta_t) * sin_i, cos_t = sqrt(1 - sin_t^2)
    let sin_i = (1.0 - cos_i * cos_i).sqrt();
    let sin_t = (eta_i / eta_t) * sin_i;
    let cos_t = (1.0 - sin_t * sin_t).sqrt();

    let inputs = [
        Vec4f([cos_i, eta_t, eta_i, 0.0]),  // air -> glass at cos_i
        Vec4f([cos_t, eta_i, eta_t, 0.0]),  // glass -> air at cos_t
    ];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 2);

    // These should be equal by reciprocity
    let r_forward = outputs[0].0[0];
    let r_backward = outputs[1].0[0];
    assert!(
        (r_forward - r_backward).abs() < 0.02,
        "Reciprocity violated: R(air->glass, cos_i={cos_i}) = {r_forward}, R(glass->air, cos_t={cos_t}) = {r_backward}"
    );
}

#[test]
fn exact_fresnel_glass_exit_normal_incidence() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // Production ordering: (cos_i, eta_i, eta_t, _).
    let inputs = [Vec4f([1.0, 1.5, 1.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_fresnel_dielectric", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.04, GPU_TOLERANCE);
}

#[test]
fn exact_fresnel_equal_media_has_no_reflection() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let inputs = [Vec4f([0.5, 1.33, 1.33, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_fresnel_dielectric", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.0, GPU_TOLERANCE);
}

// --- Schlick at Brewster's angle ---
// At Brewster's angle, the exact Fresnel r_parallel = 0 (only perpendicular reflects).
// Average reflectance should be low. Schlick won't be zero but should be small.

#[test]
fn schlick_at_brewster_angle() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // Brewster's angle for air->glass: tan(theta_B) = eta_t/eta_i = 1.5
    let eta_i = 1.0_f32;
    let eta_t = 1.5_f32;
    let theta_b = (eta_t / eta_i).atan();
    let cos_b = theta_b.cos();

    let inputs = [Vec4f([cos_b, eta_t, eta_i, 0.0])];
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_full", &inputs, 1);

    let exact = fresnel_dielectric(cos_b, eta_i, eta_t);
    let schlick_val = outputs[0].0[0];

    // Both should be low at Brewster's angle (~0.04 for glass)
    assert!(
        schlick_val < 0.15,
        "Schlick at Brewster's angle = {schlick_val}, should be low"
    );
    assert!(
        (schlick_val - exact).abs() < 0.05,
        "Schlick at Brewster's = {schlick_val}, exact = {exact}"
    );
}

// --- schlick_lazanyi: sweep cos_i, verify monotonic increase toward grazing ---

#[test]
fn schlick_lazanyi_monotonic_increase() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    // k=0: reflectance should increase monotonically as cos_i decreases (grazing)
    let eta_t = 1.5_f32;
    let cos_values: Vec<f32> = (0..10).map(|i| 1.0 - i as f32 * 0.1).collect();
    let inputs: Vec<Vec4f> = cos_values.iter().map(|&c| Vec4f([c, eta_t, 0.0, 0.0])).collect();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_lazanyi", &inputs, 10);

    for i in 1..outputs.len() {
        assert!(
            outputs[i].0[0] >= outputs[i - 1].0[0] - GPU_TOLERANCE,
            "schlick_lazanyi not monotonic: R(cos_i={:.1}) = {} < R(cos_i={:.1}) = {}",
            cos_values[i], outputs[i].0[0], cos_values[i - 1], outputs[i - 1].0[0]
        );
    }
}

// --- schlick_lazanyi: verify against CPU reference at many angles ---

#[test]
fn schlick_lazanyi_exact_match() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_fresnel_module(&ctx, &mut compiler);

    let test_cases: Vec<(f32, f32, f32)> = vec![
        (1.0, 1.5, 0.0),
        (0.5, 1.5, 0.0),
        (0.0, 1.5, 0.0),
        (0.7, 2.42, 0.0),  // diamond
        (0.3, 1.5, 1.0),   // high absorption
        (0.9, 1.33, 0.1),  // water with slight absorption
        (0.1, 3.0, 2.0),   // metal-like (high k)
    ];
    let inputs: Vec<Vec4f> = test_cases.iter().map(|&(c, e, k)| Vec4f([c, e, k, 0.0])).collect();
    let n = inputs.len();
    let outputs: Vec<Vec4f> = run_test_shader(&ctx, &module, "test_schlick_lazanyi", &inputs, n);

    for (i, &(cos_i, eta_t, k)) in test_cases.iter().enumerate() {
        let expected = cpu_schlick_lazanyi(cos_i, eta_t, k);
        assert_approx_eq(outputs[i].0[0], expected, GPU_TOLERANCE);
    }
}
