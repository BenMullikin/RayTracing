use std::sync::Arc;
use winit::window::Window;

use crate::engine::buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SystemUniform {
    pub resolution: [u32; 2],
    pub time: f32,
    pub dt: f32,
    pub frame: u32,
    pub tick: u32,
    _padding: [u32; 2],
}

impl SystemUniform {
    fn new() -> Self {
        Self {
            resolution: [0, 0],
            time: 0.0,
            dt: 0.0,
            frame: 0,
            tick: 0,
            _padding: [0; 2],
        }
    }
}

pub struct Engine {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    pub is_surface_configured: bool,
    pub system_uniform: SystemUniform,
    pub system_uniform_buffer: wgpu::Buffer,
    pub system_bind_group_layout: wgpu::BindGroupLayout,
    pub system_bind_group: wgpu::BindGroup,
    pub input: crate::engine::input::Input,
}

impl Engine {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let mut size = window.inner_size();

        size.width = size.width.max(1);
        size.height = size.height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let required_limits = if cfg!(target_arch = "wasm32") {
            let supported_limits = adapter.limits();

            wgpu::Limits {
                max_compute_workgroup_storage_size: supported_limits
                    .max_compute_workgroup_storage_size,
                max_compute_invocations_per_workgroup: supported_limits
                    .max_compute_invocations_per_workgroup,
                max_compute_workgroup_size_x: supported_limits.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: supported_limits.max_compute_workgroup_size_y,
                max_compute_workgroup_size_z: supported_limits.max_compute_workgroup_size_z,
                max_compute_workgroups_per_dimension: supported_limits
                    .max_compute_workgroups_per_dimension,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            }
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let system_uniform = SystemUniform::new();

        let system_uniform_buffer =
            buffer::create_uniform_buffer(&device, "System Uniform Buffer", &system_uniform);

        let system_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("System Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let system_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("System Bind Group"),
            layout: &system_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: system_uniform_buffer.as_entire_binding(),
            }],
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            window,
            is_surface_configured: true,
            system_uniform,
            system_uniform_buffer,
            system_bind_group_layout,
            system_bind_group,
            input: crate::engine::input::Input::new(),
        })
    }

    pub fn update(&mut self, dt: f32) {
        self.system_uniform.dt = dt;
        self.system_uniform.time += dt;
        self.system_uniform.tick += 1;
        self.system_uniform.resolution = [self.config.width, self.config.height];

        self.queue.write_buffer(
            &self.system_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.system_uniform]),
        );
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let safe_width = width.max(1).min(max_dim);
        let safe_height = height.max(1).min(max_dim);

        if safe_width != self.config.width || safe_height != self.config.height {
            self.config.width = safe_width;
            self.config.height = safe_height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn get_current_texture(&mut self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        match self.surface.get_current_texture() {
            Ok(surface_texture) => Ok(surface_texture),
            // Outdated or Lost? Reconfigure it
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                Err(wgpu::SurfaceError::Outdated)
            }
            // Out of Memory? Crash gracefully
            Err(wgpu::SurfaceError::OutOfMemory) => Err(wgpu::SurfaceError::OutOfMemory),
            // Timeout or Other? Just skip it
            Err(wgpu::SurfaceError::Timeout) | Err(wgpu::SurfaceError::Other) => {
                Err(wgpu::SurfaceError::Timeout)
            }
        }
    }
}
