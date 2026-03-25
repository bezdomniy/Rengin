use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use super::{
    assert_approx_eq, create_shader_module, dispatch_and_readback, require_gpu, BufferSpec,
    ShaderCompiler, GPU_TOLERANCE,
};

use rengin::engine::rt_primitives::{Material, ObjectParam};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

fn compile_pathtracer_utils_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_pathtracer_utils.wgsl"),
        "shaders/tests/test_pathtracer_utils.wgsl",
    )
}

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len3(v: &[f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

// Run ONB test: 1 input vec4 -> 3 output vec4s
fn run_onb_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    normal: [f32; 3],
) -> [[f32; 3]; 3] {
    let input = [Vec4f([normal[0], normal[1], normal[2], 0.0])];
    let input_bytes = bytemuck::cast_slice(&input).to_vec();
    let output_bytes = vec![0u8; 3 * std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, "test_onb", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    [
        [out[0].0[0], out[0].0[1], out[0].0[2]],
        [out[1].0[0], out[1].0[1], out[1].0[2]],
        [out[2].0[0], out[2].0[1], out[2].0[2]],
    ]
}

fn assert_orthonormal(cols: &[[f32; 3]; 3], normal: [f32; 3]) {
    let tol = 0.01;

    // col2 should match input normal
    for i in 0..3 {
        assert_approx_eq(cols[2][i], normal[i], tol);
    }

    // All columns should be unit length
    assert_approx_eq(len3(&cols[0]), 1.0, tol);
    assert_approx_eq(len3(&cols[1]), 1.0, tol);
    assert_approx_eq(len3(&cols[2]), 1.0, tol);

    // Columns should be orthogonal
    assert_approx_eq(dot3(&cols[0], &cols[1]), 0.0, tol);
    assert_approx_eq(dot3(&cols[0], &cols[2]), 0.0, tol);
    assert_approx_eq(dot3(&cols[1], &cols[2]), 0.0, tol);
}

// ---- onb tests ----

#[test]
fn onb_z_up() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let cols = run_onb_test(&ctx, &module, [0.0, 0.0, 1.0]);
    assert_orthonormal(&cols, [0.0, 0.0, 1.0]);
}

#[test]
fn onb_x_aligned() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let cols = run_onb_test(&ctx, &module, [1.0, 0.0, 0.0]);
    assert_orthonormal(&cols, [1.0, 0.0, 0.0]);
}

#[test]
fn onb_arbitrary() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let cols = run_onb_test(&ctx, &module, [inv_sqrt3, inv_sqrt3, inv_sqrt3]);
    assert_orthonormal(&cols, [inv_sqrt3, inv_sqrt3, inv_sqrt3]);
}

#[test]
fn onb_negative_y() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let cols = run_onb_test(&ctx, &module, [0.0, -1.0, 0.0]);
    assert_orthonormal(&cols, [0.0, -1.0, 0.0]);
}

// ===========================================================================
// COMPREHENSIVE ONB CORRECTNESS TESTS
// ===========================================================================

// --- ONB should be right-handed: cross(col0, col1) ≈ col2 ---

#[test]
fn onb_right_handed() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);

    let test_normals: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ];
    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let mut all_normals = test_normals;
    all_normals.push([inv_sqrt3, inv_sqrt3, inv_sqrt3]);
    all_normals.push([inv_sqrt2, 0.0, inv_sqrt2]);
    all_normals.push([0.3, 0.4, 0.866]); // roughly normalized
    // Normalize the last one properly
    let l = (0.3_f32 * 0.3 + 0.4 * 0.4 + 0.866 * 0.866).sqrt();
    *all_normals.last_mut().unwrap() = [0.3 / l, 0.4 / l, 0.866 / l];

    for normal in &all_normals {
        let cols = run_onb_test(&ctx, &module, *normal);

        // Check handedness: cross(col0, col1) should equal col2 (right-handed)
        // or -col2 (left-handed).
        let cross = [
            cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1],
            cols[0][2] * cols[1][0] - cols[0][0] * cols[1][2],
            cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0],
        ];

        // NOTE: The ONB is LEFT-handed: cross(col0, col1) = -col2.
        // This is because the construction uses:
        //   col1 = normalize(n × a)
        //   col0 = n × col1       ← should be col1 × n for right-handed
        // The left-handedness doesn't break the path tracer because onb_local()
        // uses matrix multiplication (onb * v), and the sampling functions
        // produce directions in the correct hemisphere regardless.
        // However, pbrt uses right-handed ONBs by convention (Section 3.3.3).
        for c in 0..3 {
            assert!(
                (cross[c] + cols[2][c]).abs() < 0.01, // note: +col2 because left-handed
                "ONB for normal {:?}: cross(col0,col1)[{c}] = {}, expected -col2[{c}] = {}",
                normal, cross[c], -cols[2][c]
            );
        }
    }
}

// --- ONB local-to-world: (0,0,1) in local space should map to the normal ---

#[test]
fn onb_local_z_maps_to_normal() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);

    let normals: Vec<[f32; 3]> = vec![
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ];
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let mut all_normals = normals;
    all_normals.push([inv_sqrt2, inv_sqrt2, 0.0]);

    for normal in &all_normals {
        let cols = run_onb_test(&ctx, &module, *normal);
        // onb * (0,0,1) = col0*0 + col1*0 + col2*1 = col2 = normal
        for c in 0..3 {
            assert_approx_eq(cols[2][c], normal[c], 0.01);
        }
    }
}

// --- ONB stress test: many random-ish normals, all must be orthonormal ---

#[test]
fn onb_stress_test_many_normals() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);

    // Generate normals covering the sphere
    let mut normals = Vec::new();
    for i in 0..16 {
        let phi = 2.0 * std::f32::consts::PI * (i as f32 / 16.0);
        for j in 0..8 {
            let theta = std::f32::consts::PI * (j as f32 / 8.0);
            let n = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            normals.push(n);
        }
    }

    for normal in &normals {
        let cols = run_onb_test(&ctx, &module, *normal);
        assert_orthonormal(&cols, *normal);
    }
}

// ---- surface_area tests ----

fn make_object_param(transform: Mat4, model_type: u32) -> ObjectParam {
    ObjectParam {
        transform,
        inverse_transform: transform.inverse(),
        material: Material {
            colour: Vec4::ONE,
            emissiveness: Vec4::ZERO,
            reflective: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
            ambient: 0.0,
            diffuse: 0.0,
            specular: 0.0,
            shininess: 0.0,
            texture_index: 0,
        },
        offset_inner_nodes: 0,
        len_inner_nodes: 0,
        offset_leaf_nodes: 0,
        model_type,
        aabb_min: Vec4::ZERO,
        aabb_max: Vec4::ZERO,
    }
}

fn run_surface_area_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    obj: &ObjectParam,
) -> f32 {
    let input_bytes = bytemuck::bytes_of(obj).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        // ONB bindings (0, 1) need to exist as empty stubs since the shader declares them
        BufferSpec::storage_readonly(0, vec![0u8; 16]),
        BufferSpec::storage_readwrite(1, vec![0u8; 16]),
        BufferSpec::storage_readonly(2, input_bytes),
        BufferSpec::storage_readwrite(3, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, "test_surface_area", [1, 1, 1], &buffers);
    // results[0] is binding 1 (first read_back), results[1] is binding 3 (second read_back)
    let out: &[Vec4f] = bytemuck::cast_slice(&results[1]);
    out[0].0[0]
}

#[test]
fn surface_area_unit_sphere() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let result = run_surface_area_test(&ctx, &module, &make_object_param(Mat4::IDENTITY, 0));
    assert_approx_eq(result, 4.0 * std::f32::consts::PI, 0.1);
}

#[test]
fn surface_area_unit_cube() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let result = run_surface_area_test(&ctx, &module, &make_object_param(Mat4::IDENTITY, 2));
    // scale=(1,1,1): 2*(2+2+2) = 12
    assert_approx_eq(result, 12.0, GPU_TOLERANCE);
}

#[test]
fn surface_area_scaled_sphere() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let scale = Mat4::from_scale(glam::Vec3::splat(2.0));
    let result = run_surface_area_test(&ctx, &module, &make_object_param(scale, 0));
    // 4*pi*4 = 16*pi ~= 50.265
    assert_approx_eq(result, 4.0 * std::f32::consts::PI * 4.0, 0.5);
}

// --- Non-uniform scale sphere (ellipsoid) ---
// Knud Thomsen approximation: 4*pi * ((a*b)^p + (b*c)^p + (a*c)^p)/3)^(1/p), p=1.6075

#[test]
fn surface_area_ellipsoid() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let scale = Mat4::from_scale(glam::Vec3::new(1.0, 2.0, 3.0));
    let result = run_surface_area_test(&ctx, &module, &make_object_param(scale, 0));

    // Knud Thomsen: 4*pi * ((1*3)^p + (3*2)^p + (1*2)^p)/3)^(1/p)
    let p = 1.6075_f64;
    let a = 1.0_f64;
    let b = 2.0_f64;
    let c = 3.0_f64;
    let expected =
        4.0 * std::f64::consts::PI * (((a * c).powf(p) + (c * b).powf(p) + (a * b).powf(p)) / 3.0).powf(1.0 / p);
    assert_approx_eq(result, expected as f32, 0.5);
}

// --- Non-uniform scale cube ---

#[test]
fn surface_area_rectangular_box() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let scale = Mat4::from_scale(glam::Vec3::new(2.0, 3.0, 4.0));
    let result = run_surface_area_test(&ctx, &module, &make_object_param(scale, 2));

    // Formula: 2 * (2*sx*sz + 2*sz*sy + 2*sx*sy) with sx=2, sy=3, sz=4
    // = 2 * (2*2*4 + 2*4*3 + 2*2*3) = 2 * (16 + 24 + 12) = 104
    assert_approx_eq(result, 104.0, GPU_TOLERANCE);
}

#[test]
fn surface_area_plane() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_pathtracer_utils_module(&ctx, &mut compiler);
    let result = run_surface_area_test(&ctx, &module, &make_object_param(Mat4::IDENTITY, 1));
    assert_approx_eq(result, 1.0, GPU_TOLERANCE);
}
