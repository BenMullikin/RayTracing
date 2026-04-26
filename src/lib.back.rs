use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    frame: u32,
    time: f32,
    zoom: f32,
    _padding_1: f32,
    pan: [f32; 2],
    cursor_position: [f32; 2],
    cursor_button: f32,
    _padding_2: f32,
}

impl Uniform {
    fn new() -> Self {
        Self {
            frame: 0,
            time: 0.0,
            zoom: 1.0,
            pan: [0.0, 0.0],
            _padding_1: 0.0,
            cursor_position: [0.0, 0.0],
            cursor_button: 0.0,
            _padding_2: 0.0,
        }
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,
    render_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    uniform: Uniform,
    uniform_buffer: wgpu::Buffer,
    window: Arc<Window>,
    keys_pressed: std::collections::HashSet<KeyCode>,
    mouse_pressed: std::collections::HashSet<MouseButton>,
    mouse_scroll: f32,
    cursor_position: [f32; 2],
    cursor_last_position: [f32; 2],
    last_render_time: web_time::Instant,
    frames: u32,
    fps_timer: f32,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let mut size = window.inner_size();

        size.width = size.width.max(1);
        size.height = size.height.max(1);

        // The instance is a handle to our GPU
        // BackendBit::Primary => Vulkan + Metal + DX12 + Browser WebGPU
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
            // Get the maximum limits the adapter ACTUALLY supports
            let supported_limits = adapter.limits();

            // We can manually enforce that we need compute capabilities,
            // but otherwise accept whatever the browser gives us.
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
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we need to disable some.
                required_limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
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

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    // ReadOnly Bind Group
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // WriteOnly Bind Group
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Uniform Bind Group
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let uniform = Uniform::new();

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (render_bind_group_a, render_bind_group_b, compute_bind_group_a, compute_bind_group_b) =
            Self::create_render_target(
                config.width,
                config.height,
                &device,
                &render_bind_group_layout,
                &compute_bind_group_layout,
                &uniform_buffer,
            );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("turing-shader-lighted.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let last_render_time = web_time::Instant::now();
        let frames = 0;
        let fps_timer = 0.0;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: true,
            render_pipeline,
            render_bind_group_a,
            render_bind_group_b,
            render_bind_group_layout,
            compute_pipeline,
            compute_bind_group_a,
            compute_bind_group_b,
            compute_bind_group_layout,
            uniform,
            uniform_buffer,
            window,
            keys_pressed: std::collections::HashSet::new(),
            mouse_pressed: std::collections::HashSet::new(),
            mouse_scroll: 0.0,
            cursor_position: [0.0, 0.0],
            cursor_last_position: [0.0, 0.0],
            last_render_time,
            frames,
            fps_timer,
        })
    }

    fn create_render_target(
        width: u32,
        height: u32,
        device: &wgpu::Device,
        render_layout: &wgpu::BindGroupLayout,
        compute_layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
    ) -> (
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let target_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raytracer Target Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let target_view_a = target_a.create_view(&wgpu::TextureViewDescriptor::default());

        let target_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raytracer Target Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let target_view_b = target_b.create_view(&wgpu::TextureViewDescriptor::default());

        let render_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&target_view_a),
            }],
        });

        let render_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&target_view_b),
            }],
        });

        let compute_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&target_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&target_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        (
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let safe_width = width.min(max_dim);
        let safe_height = height.min(max_dim);

        if safe_width > 0 && safe_height > 0 {
            //if safe_width == self.config.width && safe_height == self.config.height {
            //  return;
            //}

            self.config.width = safe_width;
            self.config.height = safe_height;
            self.surface.configure(&self.device, &self.config);

            let (
                render_bind_group_a,
                render_bind_group_b,
                compute_bind_group_a,
                compute_bind_group_b,
            ) = Self::create_render_target(
                safe_width,
                safe_height,
                &self.device,
                &self.render_bind_group_layout,
                &self.compute_bind_group_layout,
                &self.uniform_buffer,
            );
            self.render_bind_group_a = render_bind_group_a;
            self.render_bind_group_b = render_bind_group_b;
            self.compute_bind_group_a = compute_bind_group_a;
            self.compute_bind_group_b = compute_bind_group_b;

            self.is_surface_configured = true;
        }
    }

    fn update(&mut self) {
        self.uniform.frame += 1;
        let now = web_time::Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32();
        self.last_render_time = now;

        self.uniform.time += dt;

        let speed = 0.5 * dt * self.uniform.zoom;
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.uniform.pan[1] -= speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.uniform.pan[1] += speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.uniform.pan[0] -= speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.uniform.pan[0] += speed;
        }
        if self.keys_pressed.contains(&KeyCode::ArrowUp) {
            self.uniform.zoom *= 1.01;
        }
        if self.keys_pressed.contains(&KeyCode::ArrowDown) {
            self.uniform.zoom *= 0.99;
        }

        if self.keys_pressed.contains(&KeyCode::ArrowLeft) {
            self.uniform.time -= speed;
        }
        if self.keys_pressed.contains(&KeyCode::ArrowRight) {
            self.uniform.time += speed;
        }

        let aspect = self.config.width as f32 / self.config.height as f32;

        let uv_x = self.cursor_position[0] * aspect;
        let uv_y = self.cursor_position[1];

        if self.mouse_scroll != 0.0 {
            let old_zoom = self.uniform.zoom;

            if self.mouse_scroll > 0.0 {
                self.uniform.zoom *= 0.9;
            } else {
                self.uniform.zoom *= 1.1;
            }

            self.uniform.pan[0] += uv_x * 1.5 * (old_zoom - self.uniform.zoom);
            self.uniform.pan[1] += uv_y * 1.5 * (old_zoom - self.uniform.zoom);

            self.mouse_scroll = 0.0;
        }

        if self.mouse_pressed.contains(&MouseButton::Left) {
            let dx = uv_x - self.cursor_position[0];
            let dy = uv_y - self.cursor_position[1];

            self.uniform.pan[0] += dx * 1.5 * self.uniform.zoom;
            self.uniform.pan[1] += dy * 1.5 * self.uniform.zoom;
        }

        self.uniform.cursor_position = self.cursor_position;
        self.uniform.cursor_button = if self.mouse_pressed.contains(&MouseButton::Left) {
            1.0
        } else {
            0.0
        };

        self.cursor_last_position = self.cursor_position;

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        self.frames += 1;
        self.fps_timer += dt;

        if self.fps_timer >= 0.5 {
            let fps = self.frames as f32 / self.fps_timer;
            log::info!("Engine FPS: {:.1}", fps);
            println!("Engine FPS: {:.1}", fps);

            self.frames = 0;
            self.fps_timer = 0.0;
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if is_pressed {
            self.keys_pressed.insert(code);
        } else {
            self.keys_pressed.remove(&code);
        }

        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.window.focus_window();

        if state == ElementState::Pressed {
            self.mouse_pressed.insert(button);
        } else {
            self.mouse_pressed.remove(&button);
        }
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.mouse_scroll = -match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => scroll as f32,
        };
    }

    fn handle_cursor_position(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_position = [
            (position.x as f32 / self.config.width as f32) * 2.0 - 1.0,
            (position.y as f32 / self.config.height as f32) * 2.0 - 1.0,
        ];
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        // We have to have a configured surface to render
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            Ok(surface_texture) => surface_texture,
            // Outdated or Lost? Reconfigure it
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            // Out of Memory? Crash gracefully
            Err(wgpu::SurfaceError::OutOfMemory) => {
                anyhow::bail!("Out of Memory while aquiring surface texture");
            }
            // Timeout or Other? Just skip it
            Err(wgpu::SurfaceError::Timeout) | Err(wgpu::SurfaceError::Other) => {
                return Ok(());
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let sim_steps = 10;
        for i in 0..sim_steps {
            let step_count = self.frames * sim_steps + i;
            let current_compute_group = if step_count.is_multiple_of(2) {
                &self.compute_bind_group_a
            } else {
                &self.compute_bind_group_b
            };

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, current_compute_group, &[]);

            let workgroup_x = self.config.width.div_ceil(16);
            let workgroup_y = self.config.height.div_ceil(16);

            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        let current_render_group = if (self.frames * sim_steps + sim_steps).is_multiple_of(2) {
            &self.render_bind_group_a
        } else {
            &self.render_bind_group_b
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            // This will be a bind group (E.G. render_pass.set_bind_group(...))
            render_pass.set_bind_group(0, current_render_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element: web_sys::HtmlCanvasElement = canvas.unchecked_into();

            html_canvas_element.set_tab_index(0);
            let _ = html_canvas_element.focus();

            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on the web we can use pollster to await the
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the Future async and use the proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(
                                State::new(window)
                                    .await
                                    .expect("Unable to create cavnas!!!")
                            )
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let stat = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => stat.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                stat.update();
                match stat.render() {
                    Ok(_) => {}
                    Err(e) => {
                        // Log the error gracefully
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => stat.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseInput { state, button, .. } => {
                stat.handle_mouse_button(button, state)
            }
            WindowEvent::MouseWheel { delta, .. } => stat.handle_mouse_wheel(delta),
            WindowEvent::CursorMoved { position, .. } => stat.handle_cursor_position(position),
            WindowEvent::Focused { .. } => {
                log::info!("Focused")
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new();
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let app = App::new(&event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
