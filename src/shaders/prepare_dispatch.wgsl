// Prepare indirect dispatch args from alive ray count.
// Swaps: input_count = output_count, output_count = 0

@group(0) @binding(0)
var<storage, read_write> ray_counts: array<atomic<u32>>;
// ray_counts[0] = input count (how many rays to process this bounce)
// ray_counts[1] = output count (how many alive rays were written)

@group(0) @binding(1)
var<storage, read_write> indirect_args: array<u32>;
// indirect_args = [workgroups_x, 1, 1]

@compute @workgroup_size(1)
fn main() {
    let alive = atomicLoad(&ray_counts[1]);
    indirect_args[0] = (alive + 255u) / 256u;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;

    // Swap: next bounce reads what this bounce wrote
    atomicStore(&ray_counts[0], alive);
    atomicStore(&ray_counts[1], 0u);
}
