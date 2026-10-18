use bytemuck::{Pod, Zeroable};

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

fn compile_refraction_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_refraction.wgsl"),
        "shaders/tests/test_refraction.wgsl",
    )
}

fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// Helper: CPU Snell's law refraction direction (pbrt 9.3)
fn cpu_refract(eyev: [f32; 3], normalv: [f32; 3], cos_i: f32, ri: f32, eta_t: f32) -> Option<[f32; 3]> {
    let n_ratio = ri / eta_t;
    let sin_2t = n_ratio * n_ratio * (1.0 - cos_i * cos_i);
    if sin_2t > 1.0 {
        return None; // TIR
    }
    let cos_t = (1.0 - sin_2t).sqrt();
    let d = [
        normalv[0] * (n_ratio * cos_i - cos_t) - eyev[0] * n_ratio,
        normalv[1] * (n_ratio * cos_i - cos_t) - eyev[1] * n_ratio,
        normalv[2] * (n_ratio * cos_i - cos_t) - eyev[2] * n_ratio,
    ];
    let len = vec3_len(d);
    Some([d[0] / len, d[1] / len, d[2] / len])
}

fn run_snell_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    eyev: [f32; 3],
    normalv: [f32; 3],
    cos_i: f32,
    ri: f32,
    eta_t: f32,
) -> (Option<[f32; 3]>, f32) {
    let inputs = vec![
        Vec4f([eyev[0], eyev[1], eyev[2], cos_i]),
        Vec4f([normalv[0], normalv[1], normalv[2], 0.0]),
        Vec4f([ri, eta_t, 0.0, 0.0]),
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(ctx, module, "test_snell_refraction", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);

    let flag = out[0].0[3];
    if flag < 0.0 {
        (None, flag) // TIR
    } else {
        (Some([out[0].0[0], out[0].0[1], out[0].0[2]]), flag)
    }
}

// ===========================================================================
// Snell's law refraction tests (pbrt 9.3)
// ===========================================================================

#[test]
fn snell_air_to_glass_normal_incidence() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Normal incidence: eyev = (0,0,1), normalv = (0,0,1), cos_i = 1.0
    // At normal incidence, refracted ray goes straight through
    let (dir, _) = run_snell_test(&ctx, &module, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0, 1.0, 1.5);
    let dir = dir.expect("should not TIR at normal incidence");
    // Direction should be approximately (0, 0, -1) (continuing through)
    // Actually: d = n*(n_ratio*cos_i - cos_t) - eye*n_ratio
    // n_ratio = 1/1.5 = 0.667, sin2t = 0.667^2 * 0 = 0, cos_t = 1
    // d = (0,0,1)*(0.667*1 - 1) - (0,0,1)*0.667 = (0,0,-0.333) - (0,0,0.667) = (0,0,-1)
    assert!(
        (dir[2] - (-1.0)).abs() < GPU_TOLERANCE,
        "normal incidence: refracted direction should be (0,0,-1), got {:?}",
        dir
    );
}

#[test]
fn snell_air_to_glass_45_degrees() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // 45° incidence into glass (n=1.5)
    // eyev = normalized(-sin45, 0, cos45) = (-0.707, 0, 0.707) — but eyev = -rayD
    // Actually eyev points back toward camera. Let's set up correctly:
    // Ray going (sin45, 0, -cos45), eyev = (-sin45, 0, cos45)
    let sin_i = (45.0_f32).to_radians().sin();
    let cos_i = (45.0_f32).to_radians().cos();
    let eyev = [-sin_i, 0.0, cos_i]; // points back from surface
    let normalv = [0.0, 0.0, 1.0];

    let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.0, 1.5);
    let dir = dir.expect("should not TIR for air->glass at 45°");

    // Verify Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
    // sin(theta_t) = sin(45°) / 1.5 = 0.4714
    let expected_sin_t = sin_i / 1.5;
    // The refracted direction's transverse component should give sin(theta_t)
    // Since normal is (0,0,1), the transverse component is the x component
    let actual_sin_t = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    assert!(
        (actual_sin_t - expected_sin_t).abs() < 0.02,
        "Snell's law: sin(theta_t) = {actual_sin_t}, expected {expected_sin_t}"
    );
    assert_approx_eq(vec3_len(dir), 1.0, GPU_TOLERANCE);
}

#[test]
fn snell_glass_to_air_30_degrees() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Glass (1.5) to air (1.0) at 30° — should refract with wider angle
    let cos_i = (30.0_f32).to_radians().cos();
    let sin_i = (30.0_f32).to_radians().sin();
    let eyev = [-sin_i, 0.0, cos_i];
    let normalv = [0.0, 0.0, 1.0];

    let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.5, 1.0);
    let dir = dir.expect("should not TIR at 30° for glass->air");

    let expected_sin_t = 1.5 * sin_i / 1.0; // = 0.75
    let actual_sin_t = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    assert!(
        (actual_sin_t - expected_sin_t).abs() < 0.02,
        "Snell's law: sin(theta_t) = {actual_sin_t}, expected {expected_sin_t}"
    );
}

#[test]
fn total_internal_reflection_steep_angle() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Glass to air at 60° — critical angle for n=1.5 is arcsin(1/1.5) ≈ 41.8°
    // 60° > 41.8° → TIR
    let cos_i = (60.0_f32).to_radians().cos();
    let sin_i = (60.0_f32).to_radians().sin();
    let eyev = [-sin_i, 0.0, cos_i];
    let normalv = [0.0, 0.0, 1.0];

    let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.5, 1.0);
    assert!(dir.is_none(), "should be TIR at 60° for glass->air (critical angle ≈ 41.8°)");
}

#[test]
fn total_internal_reflection_near_critical() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Just above critical angle: arcsin(1/1.5) ≈ 41.81° + 1° = 42.81°
    let angle = 42.81_f32.to_radians();
    let cos_i = angle.cos();
    let sin_i = angle.sin();
    let eyev = [-sin_i, 0.0, cos_i];
    let normalv = [0.0, 0.0, 1.0];

    let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.5, 1.0);
    assert!(dir.is_none(), "should be TIR just above critical angle");
}

#[test]
fn no_tir_below_critical_angle() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Just below critical angle: 40° < 41.81°
    let angle = 40.0_f32.to_radians();
    let cos_i = angle.cos();
    let sin_i = angle.sin();
    let eyev = [-sin_i, 0.0, cos_i];
    let normalv = [0.0, 0.0, 1.0];

    let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.5, 1.0);
    assert!(dir.is_some(), "should NOT be TIR below critical angle");
}

#[test]
fn snell_refraction_matches_cpu() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Multiple test cases compared to CPU reference
    let cases = vec![
        // (eyev, normalv, cos_i, ri, eta_t)
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0_f32, 1.0_f32, 1.5_f32), // normal incidence
        ([-0.5_f32.sqrt(), 0.0, 0.5_f32.sqrt()], [0.0, 0.0, 1.0], 0.5_f32.sqrt(), 1.0, 1.33), // 45° water
        ([-0.258819, 0.0, 0.965926], [0.0, 0.0, 1.0], 0.965926_f32, 1.0, 2.42), // 15° diamond
    ];

    for (eyev, normalv, cos_i, ri, eta_t) in &cases {
        let (gpu_dir, _) = run_snell_test(&ctx, &module, *eyev, *normalv, *cos_i, *ri, *eta_t);
        let cpu_dir = cpu_refract(*eyev, *normalv, *cos_i, *ri, *eta_t);

        match (gpu_dir, cpu_dir) {
            (Some(g), Some(c)) => {
                for comp in 0..3 {
                    assert!(
                        (g[comp] - c[comp]).abs() < 0.01,
                        "Snell CPU/GPU mismatch for ri={ri}, eta_t={eta_t}, cos_i={cos_i}: GPU={:?}, CPU={:?}",
                        g, c
                    );
                }
            }
            (None, None) => {} // both TIR, ok
            _ => panic!("TIR mismatch: GPU={:?}, CPU={:?}", gpu_dir, cpu_dir),
        }
    }
}

#[test]
fn refraction_direction_is_unit_length() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    let angles = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    for &angle_deg in &angles {
        let angle = angle_deg.to_radians();
        let cos_i = angle.cos();
        let sin_i = angle.sin();
        let eyev = [-sin_i, 0.0, cos_i];
        let normalv = [0.0, 0.0, 1.0];

        let (dir, _) = run_snell_test(&ctx, &module, eyev, normalv, cos_i, 1.0, 1.5);
        if let Some(d) = dir {
            assert_approx_eq(
                vec3_len(d),
                1.0,
                GPU_TOLERANCE,
            );
        }
    }
}

// ===========================================================================
// TIR check tests
// ===========================================================================

#[test]
fn tir_check_no_tir() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Air to glass at 30°: no TIR expected
    let inputs = [Vec4f([1.0, 1.5, (30.0_f32).to_radians().cos(), 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_tir_check", &inputs, 1);
    // output[0] = is_tir (from snell_refract flag), output[1] = reflectance (from schlick_lazanyi)
    assert!(outputs[0].0[0] < 0.5, "should NOT be TIR");
    assert!(outputs[0].0[1] > 0.0, "reflectance should be positive");
}

#[test]
fn tir_check_tir() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Glass to air at 60° (above critical angle ~41.8°): TIR expected
    let inputs = [Vec4f([1.5, 1.0, (60.0_f32).to_radians().cos(), 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_tir_check", &inputs, 1);
    assert!(outputs[0].0[0] > 0.5, "should be TIR");
}

// ===========================================================================
// Russian roulette tests (pbrt 13.7)
// ===========================================================================

#[test]
fn russian_roulette_high_throughput() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // throughput = (1, 1, 1): max_component >= 1.0, no RR
    let inputs = [Vec4f([1.0, 1.0, 1.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_russian_roulette", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], -1.0, GPU_TOLERANCE); // q = -1 means no RR
}

#[test]
fn russian_roulette_low_throughput() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // throughput = (0.1, 0.1, 0.1): max = 0.1, q = min(0.5, 0.9) = 0.5
    let inputs = [Vec4f([0.1, 0.1, 0.1, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_russian_roulette", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.5, GPU_TOLERANCE); // q = 0.5
    // Adjusted throughput = 0.1 / (1 - 0.5) = 0.2
    assert_approx_eq(outputs[0].0[1], 0.2, GPU_TOLERANCE);
    assert_approx_eq(outputs[0].0[2], 0.2, GPU_TOLERANCE);
    assert_approx_eq(outputs[0].0[3], 0.2, GPU_TOLERANCE);
}

#[test]
fn russian_roulette_medium_throughput() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // throughput = (0.5, 0.3, 0.1): max = 0.5, q = min(0.5, 0.5) = 0.5
    let inputs = [Vec4f([0.5, 0.3, 0.1, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_russian_roulette", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.5, GPU_TOLERANCE);
    // Adjusted: 0.5/0.5=1.0, 0.3/0.5=0.6, 0.1/0.5=0.2
    assert_approx_eq(outputs[0].0[1], 1.0, GPU_TOLERANCE);
    assert_approx_eq(outputs[0].0[2], 0.6, GPU_TOLERANCE);
    assert_approx_eq(outputs[0].0[3], 0.2, GPU_TOLERANCE);
}

#[test]
fn russian_roulette_barely_below_one() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // throughput = (0.99, 0.5, 0.3): max = 0.99, q = min(0.5, 0.01) = 0.01
    let inputs = [Vec4f([0.99, 0.5, 0.3, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_russian_roulette", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.01, GPU_TOLERANCE); // q = 0.01
}

#[test]
fn russian_roulette_unbiased_statistical() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Statistical test: E[adjusted_throughput | survived] * P[survived] should equal original throughput
    // This verifies the Russian roulette is unbiased (pbrt 13.7)
    let throughput = [0.3_f32, 0.2, 0.1];
    let n = 4096u32;

    let float_inputs: Vec<Vec4f> = (0..n)
        .map(|_| Vec4f([throughput[0], throughput[1], throughput[2], 0.0]))
        .collect();
    let seed_inputs: Vec<Vec4u> = (0..n)
        .map(|i| Vec4u([i * 97 + 13, i * 53 + 7, i * 31 + 41, 0]))
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

    let results = dispatch_and_readback(
        &ctx,
        &module,
        "test_russian_roulette_statistical",
        [n, 1, 1],
        &buffers,
    );
    let outputs: &[Vec4f] = bytemuck::cast_slice(&results[0]);

    // Compute average contribution (survived paths contribute adjusted throughput, terminated contribute 0)
    let mut sum = [0.0_f64; 3];
    let mut survived = 0u32;
    for v in outputs.iter() {
        let s = v.0[3]; // 1.0 = survived, 0.0 = terminated
        if s > 0.5 {
            sum[0] += v.0[0] as f64;
            sum[1] += v.0[1] as f64;
            sum[2] += v.0[2] as f64;
            survived += 1;
        }
    }

    let mean = [
        sum[0] / n as f64,
        sum[1] / n as f64,
        sum[2] / n as f64,
    ];

    // E[contribution] should equal the original throughput (unbiased)
    for c in 0..3 {
        assert!(
            (mean[c] - throughput[c] as f64).abs() < 0.05,
            "RR unbiased test: E[contribution][{c}] = {}, expected {} (survived {}/{})",
            mean[c], throughput[c], survived, n
        );
    }

    // Sanity: some paths should survive and some should terminate
    assert!(survived > 0, "some paths should survive");
    assert!(survived < n, "some paths should terminate");
}

#[test]
fn russian_roulette_q_capped_at_half() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_refraction_module(&ctx, &mut compiler);

    // Very low throughput: max = 0.001, q = min(0.5, 0.999) = 0.5 (capped)
    let inputs = [Vec4f([0.001, 0.001, 0.001, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_russian_roulette", &inputs, 1);
    assert_approx_eq(outputs[0].0[0], 0.5, GPU_TOLERANCE);
}
