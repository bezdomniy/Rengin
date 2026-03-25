use super::{require_gpu, BufferSpec, dispatch_and_readback};

fn run_prepare_dispatch(ctx: &super::GpuTestContext, ray_counts: [u32; 2]) -> ([u32; 3], [u32; 2]) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prepare_dispatch"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../src/shaders/prepare_dispatch.wgsl"
            ))),
        });

    let ray_counts_bytes = bytemuck::cast_slice(&ray_counts).to_vec();
    let indirect_args_bytes = bytemuck::cast_slice(&[0u32, 0u32, 0u32]).to_vec();

    let buffers = vec![
        BufferSpec {
            binding: 0,
            data: ray_counts_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            read_back: true,
        },
        BufferSpec {
            binding: 1,
            data: indirect_args_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            read_back: true,
        },
    ];

    let results = dispatch_and_readback(ctx, &module, "main", [1, 1, 1], &buffers);

    let out_ray_counts: &[u32] = bytemuck::cast_slice(&results[0]);
    let out_indirect_args: &[u32] = bytemuck::cast_slice(&results[1]);

    (
        [out_indirect_args[0], out_indirect_args[1], out_indirect_args[2]],
        [out_ray_counts[0], out_ray_counts[1]],
    )
}

#[test]
fn prepare_dispatch_256_alive() {
    require_gpu!(ctx);
    let (args, counts) = run_prepare_dispatch(&ctx, [0, 256]);
    assert_eq!(args, [1, 1, 1]);
    assert_eq!(counts, [256, 0]);
}

#[test]
fn prepare_dispatch_257_alive() {
    require_gpu!(ctx);
    let (args, counts) = run_prepare_dispatch(&ctx, [0, 257]);
    assert_eq!(args, [2, 1, 1]);
    assert_eq!(counts, [257, 0]);
}

#[test]
fn prepare_dispatch_0_alive() {
    require_gpu!(ctx);
    let (args, counts) = run_prepare_dispatch(&ctx, [0, 0]);
    assert_eq!(args, [0, 1, 1]);
    assert_eq!(counts, [0, 0]);
}

#[test]
fn prepare_dispatch_1_alive() {
    require_gpu!(ctx);
    let (args, counts) = run_prepare_dispatch(&ctx, [0, 1]);
    assert_eq!(args, [1, 1, 1]);
    assert_eq!(counts, [1, 0]);
}

#[test]
fn prepare_dispatch_512_alive() {
    require_gpu!(ctx);
    let (args, counts) = run_prepare_dispatch(&ctx, [0, 512]);
    assert_eq!(args, [2, 1, 1]);
    assert_eq!(counts, [512, 0]);
}
