use bytemuck::{Pod, Zeroable};

use super::{
    assert_approx_eq, create_shader_module, dispatch_and_readback, require_gpu, BufferSpec,
    ShaderCompiler, GPU_TOLERANCE,
};

const MAXLEN: f32 = 10000.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vec4f([f32; 4]);

fn compile_intersections_module(
    ctx: &super::GpuTestContext,
    compiler: &mut ShaderCompiler,
) -> wgpu::ShaderModule {
    create_shader_module(
        ctx,
        compiler,
        include_str!("../../src/shaders/tests/test_intersections.wgsl"),
        "shaders/tests/test_intersections.wgsl",
    )
}

// Helper: build ray input as 3 vec4s
fn ray_input(origin: [f32; 3], direction: [f32; 3]) -> [Vec4f; 3] {
    [
        Vec4f([origin[0], origin[1], origin[2], 1.0]),      // rayO.xyz, ri
        Vec4f([direction[0], direction[1], direction[2], 0.0]), // rayD.xyz, pos
        Vec4f([1.0, 1.0, 1.0, 1.0]),                        // throughput
    ]
}

// Helper: run a ray intersection test with 1 test case (3 input vec4s, 2 output vec4s)
fn run_ray_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    origin: [f32; 3],
    direction: [f32; 3],
) -> (f32, i32) {
    let input = ray_input(origin, direction);
    let input_bytes = bytemuck::cast_slice(&input).to_vec();
    let output_bytes = vec![0u8; 2 * std::mem::size_of::<Vec4f>()]; // 2 vec4s output

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, entry_point, [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    // out[0] = (uv.x, uv.y, bitcast(id), closestT)
    let closest_t = out[0].0[3];
    let id: i32 = bytemuck::cast(out[0].0[2]);
    (closest_t, id)
}

fn is_hit(closest_t: f32) -> bool {
    closest_t < MAXLEN - 1.0
}

// ---- intersectSphere ----

#[test]
fn sphere_hit_center() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 4.0, GPU_TOLERANCE);
}

#[test]
fn sphere_miss() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 5.0, -5.0], [0.0, 0.0, 1.0]);
    assert!(!is_hit(t));
}

#[test]
fn sphere_inside() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 1.0, GPU_TOLERANCE);
}

#[test]
fn sphere_behind() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 0.0, 5.0], [0.0, 0.0, 1.0]);
    assert!(!is_hit(t));
}

// ---- intersectPlane ----

#[test]
fn plane_hit_from_above() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_plane", [0.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 5.0, GPU_TOLERANCE);
}

#[test]
fn plane_parallel() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_plane", [0.0, 5.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(!is_hit(t));
}

#[test]
fn plane_hit_from_below() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_plane", [0.0, -3.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 3.0, GPU_TOLERANCE);
}

// ---- intersectCube ----

#[test]
fn cube_hit_front() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_cube", [0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 4.0, GPU_TOLERANCE);
}

#[test]
fn cube_miss() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_cube", [3.0, 3.0, -5.0], [0.0, 0.0, 1.0]);
    assert!(!is_hit(t));
}

#[test]
fn cube_inside() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_cube", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(is_hit(t));
    assert_approx_eq(t, 1.0, GPU_TOLERANCE);
}

// ---- intersectAABB_inv ----

fn run_aabb_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    ray_o: [f32; 3],
    inv_d: [f32; 3],
    closest_t: f32,
    node_first: [f32; 3],
    node_second: [f32; 3],
) -> bool {
    let input = [
        Vec4f([ray_o[0], ray_o[1], ray_o[2], 0.0]),
        Vec4f([inv_d[0], inv_d[1], inv_d[2], closest_t]),
        Vec4f([node_first[0], node_first[1], node_first[2], f32::from_bits(0)]), // skip_ptr=0
        Vec4f([node_second[0], node_second[1], node_second[2], f32::from_bits(0)]), // prim_end=0
    ];
    let input_bytes = bytemuck::cast_slice(&input).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, "test_intersect_aabb", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    out[0].0[0] > 0.5
}

#[test]
fn aabb_hit() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(run_aabb_test(
        &ctx,
        &module,
        [0.0, 0.0, -5.0],
        [1e4, 1e4, 1.0], // inv_d for dir ~(0,0,1)
        MAXLEN,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ));
}

#[test]
fn aabb_miss() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(!run_aabb_test(
        &ctx,
        &module,
        [5.0, 5.0, -5.0],
        [1e4, 1e4, 1.0],
        MAXLEN,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ));
}

#[test]
fn aabb_beyond_closest_t() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Ray hits AABB at t~4, but closest_t=2
    assert!(!run_aabb_test(
        &ctx,
        &module,
        [0.0, 0.0, -5.0],
        [1e4, 1e4, 1.0],
        2.0,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ));
}

// ---- intersectWorldAABB ----

fn run_world_aabb_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    ray_o: [f32; 3],
    ray_d: [f32; 3],
    closest_t: f32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> bool {
    let input = [
        Vec4f([ray_o[0], ray_o[1], ray_o[2], 0.0]),
        Vec4f([ray_d[0], ray_d[1], ray_d[2], closest_t]),
        Vec4f([aabb_min[0], aabb_min[1], aabb_min[2], 0.0]),
        Vec4f([aabb_max[0], aabb_max[1], aabb_max[2], 0.0]),
    ];
    let input_bytes = bytemuck::cast_slice(&input).to_vec();
    let output_bytes = vec![0u8; std::mem::size_of::<Vec4f>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, "test_intersect_world_aabb", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    out[0].0[0] > 0.5
}

#[test]
fn world_aabb_hit() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(run_world_aabb_test(
        &ctx, &module,
        [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], MAXLEN,
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
    ));
}

// ===========================================================================
// COMPREHENSIVE CORRECTNESS TESTS (pbrt-derived)
// ===========================================================================

// --- Sphere: tangent ray (discriminant = 0, single root) ---
// Ray at y=1 along z-axis grazes unit sphere. t = sqrt(d²-1) where d=distance.
// For origin (0,1,-5), dir (0,0,1): a=1, b=2*0*0+2*0*1+2*(-5)*1=-10, c=0+1+25-1=25
// disc = 100-100 = 0, t = 5.0

#[test]
fn sphere_tangent_ray() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_sphere",
        [0.0, 1.0, -5.0],
        [0.0, 0.0, 1.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 5.0, GPU_TOLERANCE);
}

// --- Sphere: diagonal ray with analytical t ---
// Origin (0,0,-2), dir normalized (0,1,1)/sqrt(2) toward unit sphere
// In object space: a = dot(d,d) = 1, b = 2*dot(d,o) = 2*(0+0+(-2)/sqrt(2)) = -4/sqrt(2)
// c = dot(o,o)-1 = 4-1 = 3
// disc = b²-4ac = 16/2 - 12 = 8-12 = -4 → miss!
// Let's use origin (0,0,-2) dir (0, 0.1, 1) normalized:
// d_norm = (0, 0.1, 1)/sqrt(1.01) = (0, 0.0995, 0.9950)
// a = 1, b = 2*(0+0+(-2)*0.9950) = -3.9801, c = 3
// disc = 15.841 - 12 = 3.841, sqrt=1.960
// t1 = (3.980-1.960)/2 = 1.010, t2 = (3.980+1.960)/2 = 2.970

#[test]
fn sphere_diagonal_ray() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);

    let dy = 0.1_f32;
    let dz = 1.0_f32;
    let len = (dy * dy + dz * dz).sqrt();
    let dir = [0.0, dy / len, dz / len];
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_sphere",
        [0.0, 0.0, -2.0],
        dir,
    );
    assert!(is_hit(t));

    // Compute analytically
    let a = 1.0_f32;
    let b = 2.0 * (0.0 * dir[0] + 0.0 * dir[1] + (-2.0) * dir[2]);
    let c = 4.0 - 1.0;
    let disc = b * b - 4.0 * a * c;
    let expected_t = (-b - disc.sqrt()) / (2.0 * a);
    assert_approx_eq(t, expected_t, GPU_TOLERANCE);
}

// --- Sphere: ray along each axis ---

#[test]
fn sphere_hit_along_x_axis() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_sphere",
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 2.0, GPU_TOLERANCE);
}

#[test]
fn sphere_hit_along_y_axis() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_sphere",
        [0.0, -3.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 2.0, GPU_TOLERANCE);
}

// --- Sphere: near-epsilon origin (just outside, should hit) ---

#[test]
fn sphere_near_epsilon_outside() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Origin just outside unit sphere
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_sphere",
        [0.0, 0.0, -1.002],
        [0.0, 0.0, 1.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 0.002, 0.01);
}

// --- Cube: hit from each face ---

#[test]
fn cube_hit_top_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_cube",
        [0.0, 5.0, 0.0],
        [0.0, -1.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 4.0, GPU_TOLERANCE);
}

#[test]
fn cube_hit_bottom_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_cube",
        [0.0, -5.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 4.0, GPU_TOLERANCE);
}

#[test]
fn cube_hit_right_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_cube",
        [5.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 4.0, GPU_TOLERANCE);
}

// --- Cube: diagonal ray through corner region ---

#[test]
fn cube_diagonal_ray() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Ray from (-3,-3,-3) toward origin hits cube at the corner face
    let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_cube",
        [-3.0, -3.0, -3.0],
        [inv_sqrt3, inv_sqrt3, inv_sqrt3],
    );
    assert!(is_hit(t));
    // Distance from (-3,-3,-3) to cube face at -1: along diagonal, t_face = (-1-(-3))/inv_sqrt3 = 2*sqrt(3)
    let expected = 2.0 * 3.0_f32.sqrt();
    assert_approx_eq(t, expected, GPU_TOLERANCE);
}

// --- Cube: near-miss at edge ---

#[test]
fn cube_near_miss_edge() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Ray just barely misses cube edge
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_cube",
        [1.01, 1.01, -5.0],
        [0.0, 0.0, 1.0],
    );
    assert!(!is_hit(t));
}

// --- Plane: oblique angle with analytical t ---

#[test]
fn plane_oblique_45_degrees() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Origin at (0,2,0), direction (0,-1,1)/sqrt(2)
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_plane",
        [0.0, 2.0, 0.0],
        [0.0, -inv_sqrt2, inv_sqrt2],
    );
    assert!(is_hit(t));
    // t = -rayO.y / rayD.y = -2.0 / (-inv_sqrt2) = 2*sqrt(2)
    assert_approx_eq(t, 2.0 * 2.0_f32.sqrt(), GPU_TOLERANCE);
}

// --- Plane: near-epsilon height ---

#[test]
fn plane_near_epsilon_height() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Origin at height 0.002 (just above EPSILON=0.001), pointing down
    let (t, _) = run_ray_test(
        &ctx,
        &module,
        "test_intersect_plane",
        [0.0, 0.002, 0.0],
        [0.0, -1.0, 0.0],
    );
    assert!(is_hit(t));
    assert_approx_eq(t, 0.002, GPU_TOLERANCE);
}

// --- AABB: ray along exact edge (degenerate case) ---

#[test]
fn aabb_ray_along_face() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Ray slides along the face of the AABB at y=1 — should miss or just-hit depending on
    // the strict inequality implementation
    let result = run_aabb_test(
        &ctx,
        &module,
        [0.0, 1.0, -5.0],
        [1e10, 1.0 / 0.0, 1.0], // direction (0,0,1) exactly at y=1 face
        MAXLEN,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    );
    // At y=1.0 exactly, 0 * inf makes the y slab NaN, so the result is
    // implementation-defined. Just verify it doesn't crash.
    let _ = result;
}

// --- AABB: ray inside box ---

#[test]
fn aabb_ray_origin_inside() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Ray starts inside the AABB
    assert!(run_aabb_test(
        &ctx,
        &module,
        [0.0, 0.0, 0.0],    // origin inside [-1,1]^3
        [1.0, 1e4, 1e4],    // direction along +x
        MAXLEN,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ));
}

#[test]
fn world_aabb_miss() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(!run_world_aabb_test(
        &ctx, &module,
        [5.0, 5.0, -5.0], [0.0, 0.0, 1.0], MAXLEN,
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
    ));
}

// --- AABB: zero-thickness boxes ---
// BVH nodes holding only axis-aligned coplanar triangles (e.g. Sponza's column
// faces) have zero extent on one axis. Rays through them must still hit.

#[test]
fn aabb_flat_box_perpendicular_hit() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(run_aabb_test(
        &ctx,
        &module,
        [0.25, -0.5, -5.0],
        [1e4, 1e4, 1.0],
        MAXLEN,
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ));
}

#[test]
fn aabb_flat_box_oblique_hits() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Rays from several directions through points inside a flat box at x = 3.7.
    for (origin, target) in [
        ([-20.0, 1.3, 2.9], [3.7, 0.4, -0.6]),
        ([40.0, -7.0, 11.0], [3.7, -0.2, 0.9]),
        ([-3.0, 25.0, -18.0], [3.7, 0.8, 0.1]),
    ] {
        let direction = normalize(sub(target, origin));
        assert!(
            run_aabb_test(
                &ctx,
                &module,
                origin,
                direction.map(|component| 1.0 / component),
                MAXLEN,
                [3.7, -1.0, -1.0],
                [3.7, 1.0, 1.0],
            ),
            "ray from {origin:?} towards {target:?} missed"
        );
    }
}

#[test]
fn world_aabb_flat_box_hit() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    assert!(run_world_aabb_test(
        &ctx, &module,
        [0.0, 5.0, 0.3], [0.0, -1.0, 0.0], MAXLEN,
        [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0],
    ));
}

#[test]
fn flat_box_still_respects_closest_t_and_direction() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    // Box at t = 5 is beyond closest_t = 4.
    assert!(!run_aabb_test(
        &ctx, &module,
        [0.0, 0.0, -5.0], [1e4, 1e4, 1.0], 4.0,
        [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
    ));
    // Box is behind the ray.
    assert!(!run_aabb_test(
        &ctx, &module,
        [0.0, 0.0, 5.0], [1e4, 1e4, 1.0], MAXLEN,
        [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
    ));
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

// --- Triangles ---

fn run_triangle_test(
    ctx: &super::GpuTestContext,
    module: &wgpu::ShaderModule,
    vertices: [[f32; 3]; 3],
    origin: [f32; 3],
    direction: [f32; 3],
) -> (bool, [f32; 2], f32) {
    let vec4 = |v: [f32; 3]| Vec4f([v[0], v[1], v[2], 0.0]);
    let input = [vec4(vertices[0]), vec4(vertices[1]), vec4(vertices[2]), vec4(origin), vec4(direction)];
    let buffers = vec![
        BufferSpec::storage_readonly(0, bytemuck::cast_slice(&input).to_vec()),
        BufferSpec::storage_readwrite(1, vec![0u8; std::mem::size_of::<Vec4f>()]),
    ];
    let results = dispatch_and_readback(ctx, module, "test_intersect_triangle", [1, 1, 1], &buffers);
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    (out[0].0[0] > 0.5, [out[0].0[1], out[0].0[2]], out[0].0[3])
}

#[test]
fn triangle_hit_reports_barycentrics_of_b_and_c() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    // From both sides of the triangle, and with a non-unit direction.
    for (origin, direction, expected_t) in [
        ([0.25, 0.5, -2.0], [0.0, 0.0, 1.0], 2.0),
        ([0.25, 0.5, 3.0], [0.0, 0.0, -2.0], 1.5),
    ] {
        let (hit, uv, t) = run_triangle_test(&ctx, &module, vertices, origin, direction);
        assert!(hit);
        assert_approx_eq(uv[0], 0.25, GPU_TOLERANCE);
        assert_approx_eq(uv[1], 0.5, GPU_TOLERANCE);
        assert_approx_eq(t, expected_t, GPU_TOLERANCE);
    }
}

#[test]
fn triangle_misses_outside_behind_and_parallel() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);
    let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    for (origin, direction) in [
        ([0.75, 0.75, -2.0], [0.0, 0.0, 1.0]), // outside the hypotenuse
        ([0.25, 0.25, 2.0], [0.0, 0.0, 1.0]),  // triangle is behind the ray
        ([0.25, 0.25, 0.0], [1.0, 0.0, 0.0]),  // in the triangle's plane
    ] {
        let (hit, _, _) = run_triangle_test(&ctx, &module, vertices, origin, direction);
        assert!(!hit, "ray from {origin:?} along {direction:?} should miss");
    }
}

#[test]
fn triangles_sharing_an_edge_leave_no_gap() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);

    // A parallelogram abcd split along ac, at awkward non-axis-aligned
    // coordinates. Every ray is aimed at a point on the shared edge, so it
    // must hit at least one of the two triangles.
    let a = [13.37f32, -2.11, 5.03];
    let b = [17.9, -1.7, 6.2];
    let c = [16.3, 2.85, 4.1];
    let d = [a[0] + c[0] - b[0], a[1] + c[1] - b[1], a[2] + c[2] - b[2]];
    let mut seed = 0x9E37_79B9u32;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };

    let case_count = 65_536;
    let mut input = vec![
        Vec4f([a[0], a[1], a[2], 0.0]),
        Vec4f([b[0], b[1], b[2], 0.0]),
        Vec4f([c[0], c[1], c[2], 0.0]),
        Vec4f([d[0], d[1], d[2], 0.0]),
    ];
    for _ in 0..case_count {
        let s = random();
        let target = [0, 1, 2].map(|i| a[i] + s * (c[i] - a[i]));
        let direction = normalize([random() - 0.5, random() - 0.5, random() - 0.5]);
        let distance = 1.0 + 200.0 * random();
        let origin = [0, 1, 2].map(|i| target[i] - direction[i] * distance);
        input.push(Vec4f([origin[0], origin[1], origin[2], 0.0]));
        input.push(Vec4f([direction[0], direction[1], direction[2], 0.0]));
    }

    let buffers = vec![
        BufferSpec::storage_readonly(0, bytemuck::cast_slice(&input).to_vec()),
        BufferSpec::storage_readwrite(1, vec![0u8; case_count * std::mem::size_of::<Vec4f>()]),
    ];
    let results = dispatch_and_readback(
        &ctx,
        &module,
        "test_triangle_pair",
        [(case_count as u32).div_ceil(64), 1, 1],
        &buffers,
    );
    let out: &[Vec4f] = bytemuck::cast_slice(&results[0]);
    let gaps = out.iter().filter(|hits| hits.0[0] < 0.5 && hits.0[1] < 0.5).count();

    assert_eq!(gaps, 0, "{gaps} of {case_count} rays through the shared edge missed both triangles");
}

// --- Sphere: precision for distant origins ---

#[test]
fn sphere_hit_from_far_away_is_precise() {
    require_gpu!(ctx);
    let mut compiler = ShaderCompiler::new();
    let module = compile_intersections_module(&ctx, &mut compiler);

    // b*b - 4ac cancels badly here: the discriminant is 3.64 but the terms
    // are around 1.6e7.
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 0.3, -2000.0], [0.0, 0.0, 1.0]);
    let expected = 2000.0 - (1.0f64 - 0.09).sqrt();
    assert!(is_hit(t));
    assert!(
        (t as f64 - expected).abs() < 1e-3,
        "expected t = {expected}, got {t}"
    );

    // A ray that passes just outside the sphere must miss.
    let (t, _) = run_ray_test(&ctx, &module, "test_intersect_sphere", [0.0, 1.001, -2000.0], [0.0, 0.0, 1.0]);
    assert!(!is_hit(t), "ray at distance 1.001 from the centre hit at t = {t}");
}
