@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;

fn swap(i: u32, j: u32) {
    let temp = rays[i];
    rays[i] = rays[j];
    rays[j] = temp;
}

//TODO
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let n = arrayLength(&rays);

    if global_invocation_id.x > n / 2 {
        return;
    }
    swap(global_invocation_id.x,n-global_invocation_id.x);
}