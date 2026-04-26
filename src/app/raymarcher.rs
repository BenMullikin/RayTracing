use crate::app::Application;
use crate::engine::core::Engine;
use crate::engine::{buffer, camera::Camera, pipeline, texture::Texture};

pub struct RayMarcher {
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    sim_steps: u32,
    _target_a: Texture,
    _target_b: Texture,
}

impl RayMarcher {
    pub fn new(engine: &mut Engine) -> Self {
        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RayMarcher Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/ray-tracer-shader.wgsl").into(),
                ),
            });

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
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let camera = Camera::new([0.0, 0.0, 0.0]);
        let camera_buffer =
            buffer::create_uniform_buffer(&engine.device, "Camera Uniform", &camera.get_uniform());

        let render_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &engine.system_bind_group_layout,
                        &render_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

        let compute_pipeline_layout =
            engine
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipelien Layout"),
                    bind_group_layouts: &[
                        &engine.system_bind_group_layout,
                        &compute_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

        let render_pipeline = pipeline::create_quad_render_pipeline(
            &engine.device,
            &render_pipeline_layout,
            &shader,
            engine.config.format,
            "Render Pipeline",
        );

        let compute_pipeline = pipeline::create_compute_pipeline(
            &engine.device,
            &compute_pipeline_layout,
            &shader,
            "cs_main",
            "Compute Pipeline",
        );

        let (
            _target_a,
            _target_b,
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
        ) = Self::create_bind_groups(
            engine,
            &render_bind_group_layout,
            &compute_bind_group_layout,
            &camera_buffer,
        );

        Self {
            render_pipeline,
            compute_pipeline,
            render_bind_group_layout,
            compute_bind_group_layout,
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
            camera,
            camera_buffer,
            sim_steps: 1,
            _target_a,
            _target_b,
        }
    }

    fn create_bind_groups(
        engine: &Engine,
        render_bind_group_layout: &wgpu::BindGroupLayout,
        compute_bind_group_layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
    ) -> (
        Texture,
        Texture,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
        let _target_a = Texture::create_storage_texture(
            &engine.device,
            engine.config.width,
            engine.config.height,
            wgpu::TextureFormat::Rgba16Float,
            "Target A",
        );

        let _target_b = Texture::create_storage_texture(
            &engine.device,
            engine.config.width,
            engine.config.height,
            wgpu::TextureFormat::Rgba16Float,
            "Target B",
        );

        let render_bind_group_a = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&_target_a.view),
            }],
        });

        let render_bind_group_b = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group B"),
            layout: render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&_target_b.view),
            }],
        });

        let compute_bind_group_a = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A"),
            layout: compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&_target_b.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&_target_a.view),
                },
            ],
        });

        let compute_bind_group_b = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B"),
            layout: compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&_target_a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&_target_b.view),
                },
            ],
        });

        (
            _target_a,
            _target_b,
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
        )
    }
}

impl Application for RayMarcher {
    fn update(&mut self, engine: &mut Engine, dt: f32) {
        let old_position = self.camera.position;
        let old_yaw = self.camera.yaw;
        let old_pitch = self.camera.pitch;
        self.camera.update(&engine.input, dt);

        if old_position != self.camera.position
            || old_yaw != self.camera.yaw
            || old_pitch != self.camera.pitch
        {
            engine.system_uniform.frame = 0;
        } else {
            engine.system_uniform.frame += 1;
        }

        engine.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.get_uniform()]),
        );
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

        for i in 0..self.sim_steps {
            let step_count = engine.system_uniform.tick * self.sim_steps + i;
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            let current_compute_group = if step_count.is_multiple_of(2) {
                &self.compute_bind_group_a
            } else {
                &self.compute_bind_group_b
            };

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &engine.system_bind_group, &[]);
            compute_pass.set_bind_group(1, current_compute_group, &[]);

            let workgroup_x = engine.config.width.div_ceil(16);
            let workgroup_y = engine.config.height.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
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

            let current_render_group =
                if (engine.system_uniform.tick * self.sim_steps + self.sim_steps).is_multiple_of(2)
                {
                    &self.render_bind_group_a
                } else {
                    &self.render_bind_group_b
                };

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &engine.system_bind_group, &[]);
            render_pass.set_bind_group(1, current_render_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        engine.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn resize(&mut self, engine: &mut Engine, _width: u32, _height: u32) {
        let (
            _target_a,
            _target_b,
            render_bind_group_a,
            render_bind_group_b,
            compute_bind_group_a,
            compute_bind_group_b,
        ) = Self::create_bind_groups(
            engine,
            &self.render_bind_group_layout,
            &self.compute_bind_group_layout,
            &self.camera_buffer,
        );

        self._target_a = _target_a;
        self._target_b = _target_b;
        self.render_bind_group_a = render_bind_group_a;
        self.render_bind_group_b = render_bind_group_b;
        self.compute_bind_group_a = compute_bind_group_a;
        self.compute_bind_group_b = compute_bind_group_b;
    }
}
