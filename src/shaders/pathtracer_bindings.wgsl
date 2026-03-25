#define_import_path rengin::pt_bindings

#import rengin::types::{InnerNodes, LeafNodes, Normals, ObjectParams, Ray}
#import rengin::ubo::UBO

@group(0) @binding(0)
var imageData: texture_storage_2d<rgba16float,read_write>;
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
// SOA ray buffers for coalesced GPU memory access
@group(0) @binding(6)
var<storage, read_write> ray_origins: array<vec4<f32>>;    // xyz=origin, w=refractive_index
@group(0) @binding(7)
var<storage, read_write> ray_directions: array<vec4<f32>>; // xyz=direction, w=bitcast(pos)
@group(0) @binding(8)
var<storage, read_write> ray_throughputs: array<vec4<f32>>;
@group(0) @binding(9)
var<storage, read_write> shader_stats: array<atomic<u32>>;
// Active ray compaction buffers
@group(0) @binding(10)
var<storage, read> active_rays_in: array<u32>;
@group(0) @binding(11)
var<storage, read_write> active_rays_out: array<u32>;
@group(0) @binding(12)
var<storage, read_write> ray_counts: array<atomic<u32>>;
// ray_counts[0] = input count, ray_counts[1] = output count

fn load_ray(offset: u32) -> Ray {
    let o = ray_origins[offset];
    let d = ray_directions[offset];
    return Ray(o.xyz, o.w, d.xyz, bitcast<u32>(d.w), ray_throughputs[offset]);
}

fn store_ray(offset: u32, r: Ray) {
    ray_origins[offset] = vec4<f32>(r.rayO, r.refractive_index);
    ray_directions[offset] = vec4<f32>(r.rayD, bitcast<f32>(r.pos));
    ray_throughputs[offset] = r.throughput;
}

fn store_dead_ray(offset: u32) {
    ray_origins[offset] = vec4<f32>(-1f, -1f, -1f, -1f);
    ray_directions[offset] = vec4<f32>(-1f, -1f, -1f, 0f);
    ray_throughputs[offset] = vec4<f32>(-1f);
}
