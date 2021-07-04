pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/raytracer.comp"
    }
}
