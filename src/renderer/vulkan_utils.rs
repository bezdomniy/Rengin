use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::{PhysicalDevice, PhysicalDeviceType};
// use vulkano::instance::QueueFamily;

use vulkano::Version;

use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;

use vulkano::device::Queue;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::instance::debug::{self, DebugCallback};

use std::sync::Arc;

pub struct RenginVulkan {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub graphics_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
}

impl RenginVulkan {
    pub fn new(instance_extensions: &InstanceExtensions) -> RenginVulkan {
        let instance = Instance::new(
            None,
            Version::V1_2,
            instance_extensions,
            ["VK_LAYER_KHRONOS_validation"],
        )
        .unwrap();

        // // Display warnings and errors reported by the vulkan implementation
        // let debug_callback = DebugCallback::new(
        //     &instance,
        //     debug::MessageSeverity {
        //         error: true,
        //         warning: true,
        //         // performance_warning: true,
        //         information: false,
        //         // debug: false,
        //         verbose: false,
        //     },
        //     move |msg| {
        //         let level = if msg.ty.error {
        //             "ERROR"
        //         } else if msg.ty.warning || msg.ty.performance_warning {
        //             "WARNING"
        //         } else {
        //             unreachable!();
        //         };
        //         println!("{}: {}: {}", level, msg.layer_prefix, msg.description);
        //     },
        //     (),
        // )
        // .expect("failed to create debug callback");

        let physical = PhysicalDevice::enumerate(&instance)
            .min_by_key(|p| match p.properties().device_type.unwrap() {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .expect("no device available");

        let graphics_queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("couldn't find a graphical queue family");

        let compute_queue_family = physical
            .queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a graphical queue family");

        let transfer_queue_family = physical
            .queue_families()
            .find(|&q| q.explicitly_supports_transfers())
            .expect("couldn't find a transfer queue family");

        let (device, mut queues) = {
            Device::new(
                physical,
                &Features::none(),
                &DeviceExtensions::required_extensions(physical).union(&DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    ..DeviceExtensions::none()
                }),
                [
                    (compute_queue_family, 1.0),
                    (graphics_queue_family, 1.0),
                    (transfer_queue_family, 0.5),
                ]
                .iter()
                .cloned(),
            )
            .expect("failed to create device")
        };

        let compute_queue = queues.next().unwrap();
        let graphics_queue = queues.next().unwrap();
        let transfer_queue = queues.next().unwrap();

        RenginVulkan {
            instance: instance,
            device: device,
            graphics_queue: graphics_queue,
            transfer_queue: transfer_queue,
            compute_queue: compute_queue,
        }
    }
}
