// pub mod vk_utils;
pub mod wgpu_utils;

// use ash::vk;
use winit::dpi::PhysicalSize;

use crate::{
    engine::{
        bvh::Bvh,
        rt_primitives::{ObjectParam, Rays, ScreenData},
    },
    RendererType,
};

pub enum RenginShaderModule {
    WgpuShaderModule(wgpu::ShaderModule),
    // VkShaderModule(vk::ShaderModule),
}

pub trait RenginRenderer {
    fn update_window_size(&mut self, physical_size: &PhysicalSize<u32>);
    fn create_target_textures(&mut self, physical_size: &PhysicalSize<u32>);
    fn create_buffers(
        &mut self,
        bvh: &Bvh,
        screen_data: &ScreenData,
        rays: &Rays,
        object_params: &[ObjectParam],
    );
    fn create_pipelines(
        &mut self,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &Bvh,
        rays: &Rays,
        object_params: &[ObjectParam],
    );
    fn create_bind_groups(&mut self, physical_size: &PhysicalSize<u32>);
    fn create_shaders(&mut self, renderer_type: RendererType);
}
