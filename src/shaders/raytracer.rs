pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/raytracer.comp",
        vulkan_version: "1.2", spirv_version: "1.3"
    }
}
