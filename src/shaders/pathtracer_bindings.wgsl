#define_import_path rengin::pt_bindings

#import rengin::types::{InnerNodes, LeafNodes, Normals, ObjectParams, Ray}
#import rengin::ubo::UBO

@group(0) @binding(0)
var<storage, read_write> imageData: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> ubo: UBO;
@group(0) @binding(2)
var<storage, read> inner_nodes: InnerNodes;
@group(0) @binding(3)
var<storage, read> leaf_nodes: LeafNodes;
@group(0) @binding(4)
var<storage, read> normal_nodes: Normals;
@group(0) @binding(5)
var<storage, read> object_params: ObjectParams;
// Interleaved ray buffer: [origin, direction, throughput] per ray (3 vec4s each)
@group(0) @binding(6)
var<storage, read_write> ray_data: array<vec4<f32>>;
#ifdef PROFILE_STATS
@group(0) @binding(7)
var<storage, read_write> shader_stats: array<atomic<u32>>;
// With profiling: 8=in, 9=out, 10=counts
@group(0) @binding(8)
var<storage, read> active_rays_in: array<u32>;
@group(0) @binding(9)
var<storage, read_write> active_rays_out: array<u32>;
@group(0) @binding(10)
var<storage, read_write> ray_counts: array<atomic<u32>>;
#else
// Without profiling: 7=in, 8=out, 9=counts (10 storage buffers total)
@group(0) @binding(7)
var<storage, read> active_rays_in: array<u32>;
@group(0) @binding(8)
var<storage, read_write> active_rays_out: array<u32>;
@group(0) @binding(9)
var<storage, read_write> ray_counts: array<atomic<u32>>;
#endif
// ray_counts[0] = input count, ray_counts[1] = output count, ray_counts[2] = bounce depth

fn load_ray(offset: u32) -> Ray {
    let base = offset * 3u;
    let o = ray_data[base];
    let d = ray_data[base + 1u];
    return Ray(o.xyz, o.w, d.xyz, bitcast<u32>(d.w), ray_data[base + 2u]);
}

fn store_ray(offset: u32, r: Ray) {
    let base = offset * 3u;
    ray_data[base] = vec4<f32>(r.rayO, r.refractive_index);
    ray_data[base + 1u] = vec4<f32>(r.rayD, bitcast<f32>(r.pos));
    ray_data[base + 2u] = r.throughput;
}

fn store_dead_ray(offset: u32) {
    let base = offset * 3u;
    ray_data[base] = vec4<f32>(-1f, -1f, -1f, -1f);
    ray_data[base + 1u] = vec4<f32>(-1f, -1f, -1f, 0f);
    ray_data[base + 2u] = vec4<f32>(-1f);
}

fn load_ray_throughput(offset: u32) -> vec4<f32> {
    return ray_data[offset * 3u + 2u];
}
