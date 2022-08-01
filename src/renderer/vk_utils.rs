// Incomplete and not working - not a priority for now - abandoning
use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::CStr,
    io::Cursor,
    mem::{self, align_of},
    os::raw::c_char,
};

use crate::{
    engine::rt_primitives::{ObjectParam, Ray, ScreenData, Ubo},
    engine::{
        bvh::{Bvh, NodeInner, NodeLeaf, NodeNormal},
        rt_primitives::Rays,
    },
    RendererType,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::{read_spv, Align},
    vk::{
        ImageView, KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn,
        KhrPortabilitySubsetFn, PhysicalDeviceMemoryProperties, Pipeline, RenderPass,
    },
};

use ash::{vk, Entry};
pub use ash::{Device, Instance};

use bytemuck::offset_of;
use winit::{dpi::PhysicalSize, window::Window};

use super::{RenginRenderer, RenginShaderModule};

#[derive(Default, Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(
                submit_queue,
                &[submit_info.build()],
                command_buffer_reuse_fence,
            )
            .expect("queue submit failed.");
    }
}

pub struct RenginVk {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_utils_loader: DebugUtils,
    // pub window: winit::window::Window,
    // pub event_loop: RefCell<EventLoop<()>>,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub setup_command_buffer: vk::CommandBuffer,

    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub draw_commands_reuse_fence: vk::Fence,
    pub setup_commands_reuse_fence: vk::Fence,

    pub framebuffers: Vec<vk::Framebuffer>,
    pub renderpass: RenderPass,

    pub buffers: Option<HashMap<&'static str, vk::Buffer>>,
    pub buffers_memory: Option<HashMap<&'static str, vk::DeviceMemory>>,

    pub graphic_pipeline: Option<Pipeline>,
    pub compute_pipeline: Option<Pipeline>,

    pub target_image_view: Option<ImageView>,

    pub shaders: Option<HashMap<&'static str, RenginShaderModule>>,

    pub continous_motion: bool,
    pub rays_per_pixel: u32,
    pub ray_bounces: u32,
}

impl RenginVk {
    pub async fn new(
        window: &Window,
        // workgroup_size: [u32; 3],
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
    ) -> Self {
        unsafe {
            let entry = Entry::linked();
            let app_name = CStr::from_bytes_with_nul_unchecked(b"VulkanTriangle\0");

            let layer_names = [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )];
            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names = ash_window::enumerate_required_extensions(&window)
                .unwrap()
                .to_vec();
            extension_names.push(DebugUtils::name().as_ptr());

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                // Seems that this extension is not available on M1 Mac
                // extension_names.push(KhrPortabilityEnumerationFn::name().as_ptr());
                // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
                extension_names.push(KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
            }

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 0, 0));

            let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(&entry, &instance, &window, None).unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [
                Swapchain::name().as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::name().as_ptr(),
            ];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features);

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: window.inner_size().width,
                    height: window.inner_size().height,
                },
                _ => surface_capabilities.current_extent,
            };

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_resolution.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory");

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            let draw_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");
            let setup_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            record_submit_commandbuffer(
                &device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                        .image(depth_image)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1)
                                .level_count(1)
                                .build(),
                        );

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers.build()],
                    );
                },
            );

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let renderpass_attachments = [
                vk::AttachmentDescription {
                    format: surface_format.format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                vk::AttachmentDescription {
                    format: vk::Format::D16_UNORM,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];
            let color_attachment_refs = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ..Default::default()
            }];

            let subpass = vk::SubpassDescription::builder()
                .color_attachments(&color_attachment_refs)
                .depth_stencil_attachment(&depth_attachment_ref)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

            let renderpass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&renderpass_attachments)
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(&dependencies);

            let renderpass = device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap();

            let framebuffers: Vec<vk::Framebuffer> = present_image_views
                .iter()
                .map(|&present_image_view| {
                    let framebuffer_attachments = [present_image_view, depth_image_view];
                    let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(renderpass)
                        .attachments(&framebuffer_attachments)
                        .width(surface_resolution.width)
                        .height(surface_resolution.height)
                        .layers(1);

                    device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                })
                .collect();

            RenginVk {
                // event_loop: RefCell::new(event_loop),
                entry,
                instance,
                device,
                queue_family_index,
                pdevice,
                device_memory_properties,
                // window,
                surface_loader,
                surface_format,
                present_queue,
                surface_resolution,
                swapchain_loader,
                swapchain,
                present_images,
                present_image_views,
                pool,
                draw_command_buffer,
                setup_command_buffer,
                depth_image,
                depth_image_view,
                present_complete_semaphore,
                rendering_complete_semaphore,
                draw_commands_reuse_fence,
                setup_commands_reuse_fence,
                surface,
                debug_call_back,
                debug_utils_loader,
                depth_image_memory,
                framebuffers,
                renderpass,
                buffers: None,
                buffers_memory: None,
                shaders: None,
                compute_pipeline: None,
                graphic_pipeline: None,
                target_image_view: None,
                continous_motion,
                rays_per_pixel,
                ray_bounces,
            }
        }
    }

    unsafe fn create_buffer(
        &self,
        buffer_data: &[u8],
        buffer_usage_flags: vk::BufferUsageFlags,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(std::mem::size_of_val(&buffer_data) as u64)
            .usage(buffer_usage_flags) //vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = self.device.create_buffer(&buffer_info, None).unwrap();
        let buffer_memory_req = self.device.get_buffer_memory_requirements(buffer);
        let buffer_memory_index = find_memorytype_index(
            &buffer_memory_req,
            &self.device_memory_properties,
            memory_property_flags, //vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memory type for the buffer.");

        let allocate_info = vk::MemoryAllocateInfo {
            allocation_size: buffer_memory_req.size,
            memory_type_index: buffer_memory_index,
            ..Default::default()
        };
        let buffer_memory = self.device.allocate_memory(&allocate_info, None).unwrap();
        let memory_ptr = self
            .device
            .map_memory(
                buffer_memory,
                0,
                buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut slice = Align::new(memory_ptr, align_of::<u32>() as u64, buffer_memory_req.size);
        slice.copy_from_slice(&buffer_data);
        self.device.unmap_memory(buffer_memory);
        self.device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .unwrap();

        (buffer, buffer_memory)
    }

    unsafe fn create_graphics_descriptor_sets(&self) {
        // let descriptor_sizes = [
        //     vk::DescriptorPoolSize {
        //         ty: vk::DescriptorType::UNIFORM_BUFFER,
        //         descriptor_count: 1,
        //     },
        //     vk::DescriptorPoolSize {
        //         ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        //         descriptor_count: 1,
        //     },
        // ];
        // let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        //     .pool_sizes(&descriptor_sizes)
        //     .max_sets(1);

        // let descriptor_pool = self
        //     .device
        //     .create_descriptor_pool(&descriptor_pool_info, None)
        //     .unwrap();
        // let desc_layout_bindings = [
        //     vk::DescriptorSetLayoutBinding {
        //         descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        //     vk::DescriptorSetLayoutBinding {
        //         binding: 1,
        //         descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        //         descriptor_count: 1,
        //         stage_flags: vk::ShaderStageFlags::FRAGMENT,
        //         ..Default::default()
        //     },
        // ];
        // let descriptor_info =
        //     vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

        // let desc_set_layouts = [self
        //     .device
        //     .create_descriptor_set_layout(&descriptor_info, None)
        //     .unwrap()];

        // let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        //     .descriptor_pool(descriptor_pool)
        //     .set_layouts(&desc_set_layouts);
        // let descriptor_sets = self
        //     .device
        //     .allocate_descriptor_sets(&desc_alloc_info)
        //     .unwrap();

        // let uniform_color_buffer_descriptor = vk::DescriptorBufferInfo {
        //     buffer: uniform_color_buffer,
        //     offset: 0,
        //     range: mem::size_of_val(&uniform_color_buffer_data) as u64,
        // };

        // let tex_descriptor = vk::DescriptorBufferInfo {
        //     buffer: uniform_color_buffer,
        //     offset: 0,
        //     range: mem::size_of_val(&uniform_color_buffer_data) as u64,
        //     // image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        //     // image_view: tex_image_view,
        //     // sampler,
        // };

        // let write_desc_sets = [
        //     vk::WriteDescriptorSet {
        //         dst_set: descriptor_sets[0],
        //         descriptor_count: 1,
        //         descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        //         p_buffer_info: &uniform_color_buffer_descriptor,
        //         ..Default::default()
        //     },
        //     vk::WriteDescriptorSet {
        //         dst_set: descriptor_sets[0],
        //         dst_binding: 1,
        //         descriptor_count: 1,
        //         descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        //         p_image_info: &tex_descriptor,
        //         ..Default::default()
        //     },
        // ];
        // self.device.update_descriptor_sets(&write_desc_sets, &[]);

        ()
    }

    fn create_compute_descriptor_sets(&self) {
        ()
    }
}

impl Drop for RenginVk {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device
                .destroy_fence(self.draw_commands_reuse_fence, None);
            self.device
                .destroy_fence(self.setup_commands_reuse_fence, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            for &image_view in self.present_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }

            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            self.device.destroy_render_pass(self.renderpass, None);

            for (_, buffer_memory) in self.buffers_memory.as_ref().unwrap().into_iter() {
                self.device.free_memory(*buffer_memory, None);
            }
            for (_, buffer) in self.buffers.as_ref().unwrap().into_iter() {
                self.device.destroy_buffer(*buffer, None);
            }

            self.device.destroy_command_pool(self.pool, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl RenginRenderer for RenginVk {
    fn update_window_size(&mut self, physical_size: &PhysicalSize<u32>) {
        ()
    }

    fn create_target_textures(&mut self, physical_size: &PhysicalSize<u32>) {
        unsafe {
            let texture_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: self.surface_resolution.into(),
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let texture_image = self
                .device
                .create_image(&texture_create_info, None)
                .unwrap();
            let texture_memory_req = self.device.get_image_memory_requirements(texture_image);
            let texture_memory_index = find_memorytype_index(
                &texture_memory_req,
                &self.device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for target image.");

            let texture_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: texture_memory_req.size,
                memory_type_index: texture_memory_index,
                ..Default::default()
            };
            let texture_memory = self
                .device
                .allocate_memory(&texture_allocate_info, None)
                .unwrap();
            self.device
                .bind_image_memory(texture_image, texture_memory, 0)
                .expect("Unable to bind depth image memory");

            let target_image_view_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: texture_create_info.format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                image: texture_image,
                ..Default::default()
            };

            self.target_image_view = Some(
                self.device
                    .create_image_view(&target_image_view_info, None)
                    .unwrap(),
            );
        }
        ()
    }

    fn create_buffers(
        &mut self,
        bvh: &Bvh,
        screen_data: &ScreenData,
        rays: &Rays,
        object_params: &[ObjectParam],
    ) {
        let ubo = screen_data.generate_ubo();

        let screen_data_bytes = bytemuck::bytes_of(&ubo);
        let bvh_bytes: &[u8] = bytemuck::cast_slice(&bvh.inner_nodes);
        let rays_bytes: &[u8] = bytemuck::cast_slice(&rays.data);
        let object_params_bytes: &[u8] = bytemuck::cast_slice(&object_params);

        unsafe {
            self.create_buffer(
                screen_data_bytes,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            // TODO: update to use staging buffer and copy to device local memory
            self.create_buffer(
                bvh_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            self.create_buffer(
                rays_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            self.create_buffer(
                object_params_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
        }
    }

    fn create_shaders(&mut self, renderer_type: RendererType) {
        let mut compute_spv_file =
            Cursor::new(&include_bytes!("../shaders/compiled/pathtracer.spv")[..]);
        let mut vertex_spv_file =
            Cursor::new(&include_bytes!("../shaders/compiled/render.vert.spv")[..]);
        let mut frag_spv_file =
            Cursor::new(&include_bytes!("../shaders/compiled/render.frag.spv")[..]);

        let compute_code =
            read_spv(&mut compute_spv_file).expect("Failed to read compute shader spv file");
        let compute_shader_info = vk::ShaderModuleCreateInfo::builder().code(&compute_code);

        let vertex_code =
            read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

        let frag_code =
            read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

        unsafe {
            let cs_module = self
                .device
                .create_shader_module(&compute_shader_info, None)
                .expect("Compute shader module error");

            let vt_module = self
                .device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error");

            let fg_module = self
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error");

            let mut shaders = HashMap::new();
            shaders.insert("comp", RenginShaderModule::VkShaderModule(cs_module));
            shaders.insert("vert", RenginShaderModule::VkShaderModule(vt_module));
            shaders.insert("frag", RenginShaderModule::VkShaderModule(fg_module));

            self.shaders = Some(shaders);
        }
    }

    fn create_pipelines(&mut self, bvh: &Bvh, rays: &Rays, object_params: &[ObjectParam]) {
        unsafe {
            self.create_graphics_descriptor_sets();
        }
        self.create_compute_descriptor_sets();

        unsafe {
            let layout_create_info = vk::PipelineLayoutCreateInfo::default();

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap();

            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
            let shader_stage_create_infos = [
                vk::PipelineShaderStageCreateInfo {
                    module: match self.shaders.as_ref().unwrap().get("vert") {
                        Some(RenginShaderModule::VkShaderModule(m)) => *m,
                        _ => panic!("Invalid spirv vertex shader passed to render pipeline."),
                    },
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    module: match self.shaders.as_ref().unwrap().get("frag") {
                        Some(RenginShaderModule::VkShaderModule(m)) => *m,
                        _ => panic!("Invalid spirv fragment shader passed to render pipeline."),
                    },
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
            ];
            let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }];
            let vertex_input_attribute_descriptions = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, color) as u32,
                },
            ];

            let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_input_binding_descriptions);
            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.surface_resolution.width as f32,
                height: self.surface_resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [self.surface_resolution.into()];
            let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
                .scissors(&scissors)
                .viewports(&viewports);

            let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            };
            let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };
            let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            };
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states);

            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

            let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stage_create_infos)
                .vertex_input_state(&vertex_input_state_info)
                .input_assembly_state(&vertex_input_assembly_state_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_state_info)
                .depth_stencil_state(&depth_state_info)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(self.renderpass);

            // TODO
            let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder();

            let graphics_pipelines = self
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info.build()],
                    None,
                )
                .expect("Unable to create graphics pipeline");

            self.graphic_pipeline = Some(graphics_pipelines[0]);
        }
        ()
    }

    fn create_bind_groups(&mut self, physical_size: &PhysicalSize<u32>) {
        ()
    }
}
