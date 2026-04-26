use crate::app::Application;
use crate::engine::core::Engine;
use crate::engine::{buffer, pipeline, texture::Texture};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    frame: u32,                // Offset: 0
    time: f32,                 // Offset: 4
    zoom: f32,                 // Offset: 8
    _pad1: u32,                // Offset: 12 (Forces `pan` to start on an 8-byte boundary)
    pan: [f32; 2],             // Offset: 16
    cursor_position: [f32; 2], // Offset: 24
    cursor_button: f32,        // Offset: 32
    _pad2: [u32; 3],           // Offset: 36 (Pads total struct size to 48, a multiple of 16!)
    _pad3: [u32; 4],
}

impl Uniform {
    fn new() -> Self {
        Self {
            frame: 0,
            time: 0.0,
            zoom: 1.0,
            _pad1: 0,
            pan: [0.0, 0.0],
            cursor_position: [0.0, 0.0],
            cursor_button: 0.0,
            _pad2: [0; 3],
            _pad3: [0; 4],
        }
    }
}

pub struct App {
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    uniform: Uniform,
    uniform_buffer: wgpu::Buffer,
    frames: u32,
    sim_steps: u32,
    // We hold onto the textures so they aren't dropped, but we don't access them directly
    _target_a: Texture,
    _target_b: Texture,
}

impl App {
    pub fn new(engine: &mut Engine) -> Self {
        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Turing Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/ray-marcher-shader.wgsl").into(),
                ),
            });

        // 1. Layouts
        let render_bind_group_layout =
            engine
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            engine
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Compute Bind Group Layout"),
                    entries: &[
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

        // 2. Uniform Buffer
        let uniform = Uniform::new();
        let uniform_buffer =
            buffer::create_uniform_buffer(&engine.device, "Turing Uniforms", &uniform);

        // 3. Pipelines
        let render_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&render_bind_group_layout],
                    immediate_size: 0,
                });
        let render_pipeline = pipeline::create_quad_render_pipeline(
            &engine.device,
            &render_pipeline_layout,
            &shader,
            engine.config.format,
            "Turing Render Pipeline",
        );

        let compute_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    immediate_size: 0,
                });
        let compute_pipeline = pipeline::create_compute_pipeline(
            &engine.device,
            &compute_pipeline_layout,
            &shader,
            "cs_main",
            "Turing Compute Pipeline",
        );

        // 4. Initial Targets
        let (_target_a, _target_b, rbg_a, rbg_b, cbg_a, cbg_b) = Self::create_bind_groups(
            engine,
            &render_bind_group_layout,
            &compute_bind_group_layout,
            &uniform_buffer,
        );

        Self {
            render_pipeline,
            compute_pipeline,
            render_bind_group_layout,
            compute_bind_group_layout,
            render_bind_group_a: rbg_a,
            render_bind_group_b: rbg_b,
            compute_bind_group_a: cbg_a,
            compute_bind_group_b: cbg_b,
            uniform,
            uniform_buffer,
            frames: 0,
            sim_steps: 1, // Tweak this up or down for faster/slower simulation!
            _target_a,
            _target_b,
        }
    }

    fn create_bind_groups(
        engine: &Engine,
        render_layout: &wgpu::BindGroupLayout,
        compute_layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
    ) -> (
        Texture,
        Texture,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
        let target_a = Texture::create_storage_texture(
            &engine.device,
            engine.config.width,
            engine.config.height,
            wgpu::TextureFormat::Rgba16Float,
            "Target A",
        );
        let target_b = Texture::create_storage_texture(
            &engine.device,
            engine.config.width,
            engine.config.height,
            wgpu::TextureFormat::Rgba16Float,
            "Target B",
        );

        let render_bind_group_a = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&target_a.view),
            }],
        });

        let render_bind_group_b = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group B"),
            layout: render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&target_b.view),
            }],
        });

        let compute_bind_group_a = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A"),
            layout: compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target_b.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&target_a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_b = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B"),
            layout: compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target_a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&target_b.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        (
            target_a,
            target_b,
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
        )
    }
}

impl Application for App {
    fn update(&mut self, engine: &mut Engine, dt: f32) {
        self.uniform.frame += 1;
        self.uniform.time += dt;

        // Note: Assuming `engine` now has a `pub input: Input` field!
        let speed = 0.5 * dt * self.uniform.zoom;
        if engine.input.is_key_pressed(KeyCode::KeyW) {
            self.uniform.pan[1] -= speed;
        }
        if engine.input.is_key_pressed(KeyCode::KeyS) {
            self.uniform.pan[1] += speed;
        }
        if engine.input.is_key_pressed(KeyCode::KeyA) {
            self.uniform.pan[0] -= speed;
        }
        if engine.input.is_key_pressed(KeyCode::KeyD) {
            self.uniform.pan[0] += speed;
        }

        let aspect = engine.config.width as f32 / engine.config.height as f32;
        let uv_x = engine.input.cursor_position[0] * aspect;
        let uv_y = engine.input.cursor_position[1];

        // Zoom based on scroll wheel
        if engine.input.mouse_scroll != 0.0 {
            let old_zoom = self.uniform.zoom;
            if engine.input.mouse_scroll > 0.0 {
                self.uniform.zoom *= 0.9;
            } else {
                self.uniform.zoom *= 1.1;
            }

            self.uniform.pan[0] += uv_x * 1.5 * (old_zoom - self.uniform.zoom);
            self.uniform.pan[1] += uv_y * 1.5 * (old_zoom - self.uniform.zoom);
        }

        // Panning based on Mouse Drag
        if engine.input.is_mouse_pressed(MouseButton::Left) {
            let dx = engine.input.cursor_delta[0] * aspect;
            let dy = engine.input.cursor_delta[1];
            self.uniform.pan[0] -= dx * 1.5 * self.uniform.zoom;
            self.uniform.pan[1] -= dy * 1.5 * self.uniform.zoom;
        }

        self.uniform.cursor_position = [
            (engine.input.cursor_position[0] / engine.config.width as f32) * 2.0 - 1.0,
            (engine.input.cursor_position[1] / engine.config.height as f32) * 2.0 - 1.0,
        ];

        self.uniform.cursor_button = if engine.input.is_mouse_pressed(MouseButton::Left) {
            1.0
        } else {
            0.0
        };

        // Write to GPU
        engine.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );

        self.frames += 1;
    }

    fn render(&mut self, engine: &mut Engine) -> anyhow::Result<()> {
        let output = engine.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // 1. Compute Passes
        for i in 0..self.sim_steps {
            let step_count = self.frames * self.sim_steps + i;
            let current_compute_group = if step_count % 2 == 0 {
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

            let workgroup_x = (engine.config.width + 15) / 16;
            let workgroup_y = (engine.config.height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // 2. Render Pass
        let current_render_group = if (self.frames * self.sim_steps + self.sim_steps) % 2 == 0 {
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, current_render_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        engine.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Important: Reset per-frame input deltas so they don't linger!
        engine.input.finish_frame();

        Ok(())
    }

    fn resize(&mut self, engine: &mut Engine, width: u32, height: u32) {
        // Destroy old textures and create new ones that match the new window size
        let (ta, tb, ra, rb, ca, cb) = Self::create_bind_groups(
            engine,
            &self.render_bind_group_layout,
            &self.compute_bind_group_layout,
            &self.uniform_buffer,
        );

        self._target_a = ta;
        self._target_b = tb;
        self.render_bind_group_a = ra;
        self.render_bind_group_b = rb;
        self.compute_bind_group_a = ca;
        self.compute_bind_group_b = cb;
    }
}
