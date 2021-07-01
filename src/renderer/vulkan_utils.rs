use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
// use vulkano::instance::QueueFamily;

use vulkano::Version;

use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;

use vulkano::device::Queue;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use std::sync::Arc;

pub struct RenginVulkan {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub graphics_queue: Arc<Queue>, // physical: PhysicalDevice,
    // graphic_queue_family: QueueFamily
    pub compute_queue: Arc<Queue>,
}

impl RenginVulkan {
    pub fn new(instance_extensions: &InstanceExtensions) -> RenginVulkan {
        let instance = Instance::new(None, Version::V1_2, instance_extensions, None).unwrap();

        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .expect("no device available");

        let graphics_queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("couldn't find a graphical queue family");

        let compute_queue_family = physical
            .queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a graphical queue family");

        let (device, mut queues) = {
            Device::new(
                physical,
                &Features::none(),
                &DeviceExtensions::required_extensions(physical),
                [(compute_queue_family, 0.5), (graphics_queue_family, 0.5)]
                    .iter()
                    .cloned(),
            )
            .expect("failed to create device")
        };

        let graphics_queue = queues.next().unwrap();
        let compute_queue = queues.next().unwrap();

        RenginVulkan {
            instance: instance,
            device: device,
            graphics_queue: graphics_queue,
            compute_queue: compute_queue,
        }
    }
}
