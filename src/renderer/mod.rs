// pub mod vk_utils;
pub mod wgpu_utils;

// use ash::vk;
use winit::dpi::PhysicalSize;

use crate::{
    RendererType,
    engine::{
        bvh::Bvh,
        rt_primitives::{ObjectParam, ScreenData},
    },
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
        object_params: &[ObjectParam],
    );

    fn create_pipeline_layout(
        &mut self,
        pipeline_name: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bind_group_entries: &[wgpu::BindGroupEntry]
    ) -> Option<wgpu::PipelineLayout>;
    fn create_render_pipeline(
        &mut self,
        pipeline_name: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        bind_group_entries: &[wgpu::BindGroupEntry]
    ) -> Option<wgpu::RenderPipeline>;
    fn create_compute_pipeline(
        &mut self,
        pipeline_name: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        bind_group_entries: &[wgpu::BindGroupEntry]
    ) -> Option<wgpu::ComputePipeline>;

    fn create_pipelines(
        &mut self,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &Bvh,
        screen_data: &ScreenData,
        object_params: &[ObjectParam],
    );
    fn create_shaders(&mut self, renderer_type: RendererType);
}
