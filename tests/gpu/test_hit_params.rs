use bytemuck::{Pod, Zeroable};

use super::{
    assert_approx_eq, create_shader_module, dispatch_and_readback, require_gpu, run_test_shader,
    BufferSpec, ShaderCompiler, GPU_TOLERANCE,
};

/// Mirrors SELF_INTERSECTION_SCALE in hit_params_math.wgsl.
const SELF_INTERSECTION_SCALE: f32 = 64.0 * f32::EPSILON;

fn expected_offset(ray_o: [f32; 3], t: f32) -> f32 {
    let max_abs = ray_o.iter().fold(0.0f32, |acc, c| acc.max(c.abs()));
    SELF_INTERSECTION_SCALE * (max_abs + t.abs())
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

fn compile_hit_params_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_hit_params.wgsl"),
        "shaders/tests/test_hit_params.wgsl",
    )
}

fn assert_vec3_approx_eq(actual: [f32; 3], expected: [f32; 3], tol: f32, label: &str) {
    for c in 0..3 {
        assert!(
            (actual[c] - expected[c]).abs() <= tol,
            "{label} component {c}: expected {}, got {} (tol {tol})",
            expected[c], actual[c]
        );
    }
}

fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ===========================================================================
// normalAt tests
// ===========================================================================

#[test]
fn sphere_normal_at_z_pole() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Point on unit sphere at (0, 0, -1), type = 0 (sphere)
    let inputs = [Vec4f([0.0, 0.0, -1.0, 0.0])]; // type 0 = sphere
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    assert_vec3_approx_eq(
        [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]],
        [0.0, 0.0, -1.0],
        GPU_TOLERANCE,
        "sphere normal at -z pole",
    );
}

#[test]
fn sphere_normal_at_equator() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    let inputs = [Vec4f([1.0, 0.0, 0.0, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    assert_vec3_approx_eq(
        [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]],
        [1.0, 0.0, 0.0],
        GPU_TOLERANCE,
        "sphere normal at +x equator",
    );
}

#[test]
fn sphere_normal_at_diagonal() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let inputs = [Vec4f([inv_sqrt3, inv_sqrt3, inv_sqrt3, 0.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    let n = [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]];
    assert_vec3_approx_eq(n, [inv_sqrt3, inv_sqrt3, inv_sqrt3], GPU_TOLERANCE, "sphere diagonal");
    // Must be unit length
    assert_approx_eq(vec3_len(n), 1.0, GPU_TOLERANCE);
}

#[test]
fn plane_normal_always_up() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Plane (type 1) always has normal (0, 1, 0) regardless of hit point
    let inputs = vec![
        Vec4f([5.0, 0.0, 3.0, 1.0]),   // some point on plane, type=1
        Vec4f([-2.0, 0.0, 7.0, 1.0]),  // another point
        Vec4f([0.0, 0.0, 0.0, 1.0]),   // origin
    ];
    let n = inputs.len();
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, n);

    for (i, out) in outputs.iter().enumerate() {
        assert_vec3_approx_eq(
            [out.0[0], out.0[1], out.0[2]],
            [0.0, 1.0, 0.0],
            GPU_TOLERANCE,
            &format!("plane normal case {i}"),
        );
    }
}

#[test]
fn cube_normal_z_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Point on +z face: (0.5, 0.5, 1.0), type 2 = cube
    // In object space, z component is largest → normal = (0,0,1)
    let inputs = [Vec4f([0.5, 0.5, 1.0, 2.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    assert_vec3_approx_eq(
        [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]],
        [0.0, 0.0, 1.0],
        GPU_TOLERANCE,
        "cube +z face normal",
    );
}

#[test]
fn cube_normal_y_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Point on -y face: (0.3, -1.0, 0.2), type 2
    let inputs = [Vec4f([0.3, -1.0, 0.2, 2.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    assert_vec3_approx_eq(
        [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]],
        [0.0, -1.0, 0.0],
        GPU_TOLERANCE,
        "cube -y face normal",
    );
}

#[test]
fn cube_normal_x_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Point on +x face: (1.0, 0.3, -0.2), type 2
    let inputs = [Vec4f([1.0, 0.3, -0.2, 2.0])];
    let outputs: Vec<Vec4f> =
        run_test_shader(&ctx, &module, "test_normal_at", &inputs, 1);
    assert_vec3_approx_eq(
        [outputs[0].0[0], outputs[0].0[1], outputs[0].0[2]],
        [1.0, 0.0, 0.0],
        GPU_TOLERANCE,
        "cube +x face normal",
    );
}

#[test]
fn sphere_normal_with_scale() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Sphere scaled (2, 1, 1): point at (2, 0, 0) in world space
    // Object space: (1, 0, 0), normal in object = (1, 0, 0)
    // Normal in world = normalize(transpose(inv) * (1,0,0)) = normalize((1/2, 0, 0)) = (1, 0, 0)
    let inputs = vec![
        Vec4f([2.0, 0.0, 0.0, 0.0]),  // p=(2,0,0), type=sphere
        Vec4f([2.0, 1.0, 1.0, 0.0]),  // scale
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(&ctx, &module, "test_normal_at_scaled", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    let n = [out[0].0[0], out[0].0[1], out[0].0[2]];
    assert_vec3_approx_eq(n, [1.0, 0.0, 0.0], GPU_TOLERANCE, "scaled sphere normal");
    assert_approx_eq(vec3_len(n), 1.0, GPU_TOLERANCE);
}

#[test]
fn sphere_normal_with_nonuniform_scale() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Sphere scaled (1, 2, 1): point at (0, 2, 0) in world space
    // Object space: (0, 1, 0), object normal = (0, 1, 0)
    // World normal = normalize(transpose(diag(1, 1/2, 1)) * (0,1,0)) = normalize((0, 1/2, 0)) = (0, 1, 0)
    let inputs = vec![
        Vec4f([0.0, 2.0, 0.0, 0.0]),
        Vec4f([1.0, 2.0, 1.0, 0.0]),
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(&ctx, &module, "test_normal_at_scaled", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    let n = [out[0].0[0], out[0].0[1], out[0].0[2]];
    assert_vec3_approx_eq(n, [0.0, 1.0, 0.0], GPU_TOLERANCE, "nonuniform scaled sphere normal");
    assert_approx_eq(vec3_len(n), 1.0, GPU_TOLERANCE);
}

// ===========================================================================
// getHitParams tests
// ===========================================================================

fn run_hit_params_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    ray_o: [f32; 3],
    ray_d: [f32; 3],
    closest_t: f32,
    type_enum: u32,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3], bool) {
    let inputs = vec![
        Vec4f([ray_o[0], ray_o[1], ray_o[2], closest_t]),
        Vec4f([ray_d[0], ray_d[1], ray_d[2], type_enum as f32]),
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; 7 * std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(ctx, module, "test_get_hit_params", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);

    let p = [out[0].0[0], out[0].0[1], out[0].0[2]];
    let normalv = [out[1].0[0], out[1].0[1], out[1].0[2]];
    let eyev = [out[2].0[0], out[2].0[1], out[2].0[2]];
    let reflectv = [out[3].0[0], out[3].0[1], out[3].0[2]];
    let over_point = [out[4].0[0], out[4].0[1], out[4].0[2]];
    let under_point = [out[5].0[0], out[5].0[1], out[5].0[2]];
    let front_face = out[6].0[0] > 0.5;

    (p, normalv, eyev, reflectv, over_point, under_point, front_face)
}

#[test]
fn hit_params_sphere_front_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Ray from (0,0,-5) toward (0,0,1), hits unit sphere at t=4 → p=(0,0,-1)
    let (p, normalv, eyev, _reflectv, over_point, under_point, front_face) =
        run_hit_params_test(&ctx, &module, [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 4.0, 0);

    assert_vec3_approx_eq(p, [0.0, 0.0, -1.0], GPU_TOLERANCE, "hit point");
    assert_vec3_approx_eq(normalv, [0.0, 0.0, -1.0], GPU_TOLERANCE, "normal");
    assert_vec3_approx_eq(eyev, [0.0, 0.0, -1.0], GPU_TOLERANCE, "eye vector");
    assert!(front_face, "should be front face");

    // overPoint should be offset along normal (away from surface)
    assert!(
        over_point[2] < p[2],
        "overPoint.z should be more negative (along -z normal)"
    );
    // underPoint should be offset opposite to normal
    assert!(
        under_point[2] > p[2],
        "underPoint.z should be less negative (opposite normal)"
    );
}

#[test]
fn hit_params_sphere_back_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Ray from inside sphere (0,0,0) toward (0,0,1), hits at t=1 → p=(0,0,1)
    let (p, normalv, _eyev, _reflectv, _over, _under, front_face) =
        run_hit_params_test(&ctx, &module, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 0);

    assert_vec3_approx_eq(p, [0.0, 0.0, 1.0], GPU_TOLERANCE, "hit point");
    assert!(!front_face, "should be back face (ray from inside)");
    // Normal should be flipped to face the ray: (0,0,-1)
    assert_vec3_approx_eq(normalv, [0.0, 0.0, -1.0], GPU_TOLERANCE, "flipped normal");
}

#[test]
fn hit_params_plane_from_above() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Ray from (0,5,0) toward (0,-1,0), hits plane at t=5 → p=(0,0,0)
    let (p, normalv, eyev, _reflectv, over_point, under_point, front_face) =
        run_hit_params_test(&ctx, &module, [0.0, 5.0, 0.0], [0.0, -1.0, 0.0], 5.0, 1);

    assert_vec3_approx_eq(p, [0.0, 0.0, 0.0], GPU_TOLERANCE, "hit point");
    assert_vec3_approx_eq(normalv, [0.0, 1.0, 0.0], GPU_TOLERANCE, "plane normal");
    assert_vec3_approx_eq(eyev, [0.0, 1.0, 0.0], GPU_TOLERANCE, "eye");
    assert!(front_face, "should be front face");

    // overPoint/underPoint are offset along +y/-y by the scale-aware offset
    let offset = expected_offset([0.0, 5.0, 0.0], 5.0);
    assert!((over_point[1] - offset).abs() <= offset * 1e-3, "overPoint.y = {}", over_point[1]);
    assert!((under_point[1] + offset).abs() <= offset * 1e-3, "underPoint.y = {}", under_point[1]);
}

#[test]
fn hit_params_over_under_point_offset_magnitude() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // The offset scales with |ray origin| + t
    let offset = expected_offset([0.0, 0.0, -5.0], 4.0);
    let (p, normalv, _, _, over_point, under_point, _) =
        run_hit_params_test(&ctx, &module, [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 4.0, 0);

    let over_dist = vec3_len([
        over_point[0] - p[0],
        over_point[1] - p[1],
        over_point[2] - p[2],
    ]);
    assert!((over_dist - offset).abs() <= offset * 1e-2, "overPoint offset {over_dist}, expected {offset}");

    let under_dist = vec3_len([
        under_point[0] - p[0],
        under_point[1] - p[1],
        under_point[2] - p[2],
    ]);
    assert!((under_dist - offset).abs() <= offset * 1e-2, "underPoint offset {under_dist}, expected {offset}");

    // overPoint - p should be in the direction of normalv
    let over_dir = [
        over_point[0] - p[0],
        over_point[1] - p[1],
        over_point[2] - p[2],
    ];
    let dot = vec3_dot(over_dir, normalv);
    assert!(dot > 0.0, "overPoint should be in direction of normal");
}

// ===========================================================================
// reflection vector tests
// ===========================================================================

#[test]
fn reflect_normal_incidence() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // d = (0,0,1), n = (0,0,-1) → reflect = (0,0,-1)
    // reflect(d, n) = d - 2*dot(d,n)*n = (0,0,1) - 2*(-1)*(0,0,-1) = (0,0,1) + (0,0,-2) = (0,0,-1)
    let inputs = vec![
        Vec4f([0.0, 0.0, 1.0, 0.0]),  // d
        Vec4f([0.0, 0.0, -1.0, 0.0]), // n
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(&ctx, &module, "test_reflect", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    assert_vec3_approx_eq(
        [out[0].0[0], out[0].0[1], out[0].0[2]],
        [0.0, 0.0, -1.0],
        GPU_TOLERANCE,
        "reflect normal incidence",
    );
}

#[test]
fn reflect_45_degree() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // d = (1,0,-1)/sqrt(2), n = (0,0,1) → reflect = (1,0,1)/sqrt(2)
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let inputs = vec![
        Vec4f([inv_sqrt2, 0.0, -inv_sqrt2, 0.0]),
        Vec4f([0.0, 0.0, 1.0, 0.0]),
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(&ctx, &module, "test_reflect", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    assert_vec3_approx_eq(
        [out[0].0[0], out[0].0[1], out[0].0[2]],
        [inv_sqrt2, 0.0, inv_sqrt2],
        GPU_TOLERANCE,
        "reflect 45 degree",
    );
}

#[test]
fn reflect_preserves_length() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Arbitrary direction
    let d_len = (0.3_f32 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5).sqrt();
    let inputs = vec![
        Vec4f([0.3, 0.4, 0.5, 0.0]),
        Vec4f([0.0, 1.0, 0.0, 0.0]),
    ];
    let input_bytes = bytemuck::cast_slice(&inputs).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];
    let results = dispatch_and_readback(&ctx, &module, "test_reflect", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    let r_len = vec3_len([out[0].0[0], out[0].0[1], out[0].0[2]]);
    assert_approx_eq(r_len, d_len, GPU_TOLERANCE);
}

// ===========================================================================
// Self-intersection robustness
// ===========================================================================

#[test]
fn hit_params_offset_clears_plane_at_large_and_small_scales() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Rays from far-away origins and at grazing angles, plus a tiny scene
    // where a fixed 0.001 offset would be larger than the geometry.
    for (origin, target) in [
        ([4000.0, 300.0, -2500.0], [4100.0, 0.0, -2400.0]),
        ([-9000.0, 0.7, 8000.0], [-1.5, 0.0, 2.25]),
        ([0.0, 2.0, -5000.0], [3.0, 0.0, 9.0]),
        ([0.0, 0.0005, 0.0], [0.0001, 0.0, 0.0]),
    ] {
        let to_target = [
            (target[0] - origin[0]) as f64,
            (target[1] - origin[1]) as f64,
            (target[2] - origin[2]) as f64,
        ];
        let t = (to_target[0] * to_target[0] + to_target[1] * to_target[1] + to_target[2] * to_target[2]).sqrt();
        let direction = to_target.map(|c| (c / t) as f32);
        let (p, normalv, _, _, over_point, under_point, front_face) =
            run_hit_params_test(&ctx, &module, origin, direction, t as f32, 1);

        assert!(front_face, "ray from {origin:?} should hit the top of the plane");
        assert_vec3_approx_eq(normalv, [0.0, 1.0, 0.0], GPU_TOLERANCE, "plane normal");
        assert!(over_point[1] > 0.0, "overPoint {over_point:?} not above the plane (p = {p:?})");
        assert!(under_point[1] < 0.0, "underPoint {under_point:?} not below the plane (p = {p:?})");
        assert!(
            over_point[1] < 1e-3 * (1.0 + t as f32),
            "offset {} is too large for a ray of length {t}",
            over_point[1]
        );
    }
}

fn run_orient_normals_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    ray_d: [f32; 3],
    geometric_n: [f32; 3],
    shading_n: [f32; 3],
) -> ([f32; 3], [f32; 3], bool) {
    let inputs = vec![
        Vec4f([ray_d[0], ray_d[1], ray_d[2], 0.0]),
        Vec4f([geometric_n[0], geometric_n[1], geometric_n[2], 0.0]),
        Vec4f([shading_n[0], shading_n[1], shading_n[2], 0.0]),
    ];
    let buffers = vec![
        BufferSpec::storage_readonly(0, bytemuck::cast_slice(&inputs).to_vec()),
        BufferSpec::storage_readwrite(1, vec![0u8; 3 * std::mem::size_of::<Vec4f>()]),
    ];
    let results = dispatch_and_readback(ctx, module, "test_orient_normals", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    (
        [out[0].0[0], out[0].0[1], out[0].0[2]],
        [out[1].0[0], out[1].0[1], out[1].0[2]],
        out[2].0[0] > 0.5,
    )
}

#[test]
fn orient_normals_uses_geometric_side_at_grazing_angles() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // A grazing ray hits the top of a surface whose interpolated shading
    // normal is tilted towards the ray direction, so dot(ray, shading) > 0.
    let ray_d = [0.995, -0.0995, 0.0];
    let shading = [0.7, 0.714, 0.0];
    let (geometric, oriented_shading, front_face) =
        run_orient_normals_test(&ctx, &module, ray_d, [0.0, 1.0, 0.0], shading);

    assert!(front_face, "the ray hit the front of the surface");
    assert_vec3_approx_eq(geometric, [0.0, 1.0, 0.0], GPU_TOLERANCE, "geometric normal");
    assert_vec3_approx_eq(oriented_shading, shading, GPU_TOLERANCE, "shading normal");
}

#[test]
fn orient_normals_takes_outward_side_from_shading_normal() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_hit_params_module(&ctx, &mut compiler);

    // Triangle winding gives a downward geometric normal, but the vertex
    // normals point up: a ray from above hits the front.
    let (geometric, shading, front_face) =
        run_orient_normals_test(&ctx, &module, [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(front_face);
    assert_vec3_approx_eq(geometric, [0.0, 1.0, 0.0], GPU_TOLERANCE, "geometric normal");
    assert_vec3_approx_eq(shading, [0.0, 1.0, 0.0], GPU_TOLERANCE, "shading normal");

    // A ray from inside hits the back; both normals face the ray.
    let (geometric, shading, front_face) =
        run_orient_normals_test(&ctx, &module, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(!front_face);
    assert_vec3_approx_eq(geometric, [0.0, -1.0, 0.0], GPU_TOLERANCE, "geometric normal");
    assert_vec3_approx_eq(shading, [0.0, -1.0, 0.0], GPU_TOLERANCE, "shading normal");
}
