#define_import_path rengin::stats

#ifdef PROFILE_STATS
#import rengin::pt_bindings::shader_stats
var<workgroup> workgroup_stats: array<atomic<u32>, 8>;
#endif

// Shader profiling stats - atomic counters for per-frame aggregate statistics
// Stats buffer layout (8 atomic u32 counters):
//   [0] rays_alive       - number of rays that were not dead this dispatch
//   [1] world_aabb_tests - total world-space AABB tests performed
//   [2] world_aabb_skips - world-space AABB misses (object skipped entirely)
//   [3] bvh_node_tests   - total BVH inner node AABB tests
//   [4] bvh_node_skips   - BVH nodes skipped (AABB miss or t-bound prune)
//   [5] triangle_tests   - total triangle intersection tests
//   [6] triangle_hits    - triangles that produced a closer hit
//   [7] objects_tested    - objects that passed world AABB and were fully tested

var<private> p_world_aabb_tests: u32 = 0u;
var<private> p_world_aabb_skips: u32 = 0u;
var<private> p_bvh_node_tests: u32 = 0u;
var<private> p_bvh_node_skips: u32 = 0u;
var<private> p_triangle_tests: u32 = 0u;
var<private> p_triangle_hits: u32 = 0u;
var<private> p_objects_tested: u32 = 0u;

fn flush_stats(processed: bool, local_invocation_index: u32) {
#ifdef PROFILE_STATS
    if local_invocation_index < 8u {
        atomicStore(&workgroup_stats[local_invocation_index], 0u);
    }
    workgroupBarrier();

    if processed {
        atomicAdd(&workgroup_stats[0], 1u);
        atomicAdd(&workgroup_stats[1], p_world_aabb_tests);
        atomicAdd(&workgroup_stats[2], p_world_aabb_skips);
        atomicAdd(&workgroup_stats[3], p_bvh_node_tests);
        atomicAdd(&workgroup_stats[4], p_bvh_node_skips);
        atomicAdd(&workgroup_stats[5], p_triangle_tests);
        atomicAdd(&workgroup_stats[6], p_triangle_hits);
        atomicAdd(&workgroup_stats[7], p_objects_tested);
    }
    workgroupBarrier();

    if local_invocation_index < 8u {
        let value = atomicLoad(&workgroup_stats[local_invocation_index]);
        if value > 0u {
            atomicAdd(&shader_stats[local_invocation_index], value);
        }
    }
#endif
}
