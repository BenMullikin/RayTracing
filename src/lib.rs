use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use winit::event_loop::EventLoopProxy;

pub mod app;
pub mod engine;

use crate::app::Application;
use crate::engine::core::Engine;

pub enum EngineEvent<A: Application> {
    Initialized(Engine, A),
}

struct EngineRunner<A: Application + 'static> {
    engine: Option<Engine>,
    app: Option<A>,
    app_builder: Option<Box<dyn FnOnce(&mut Engine) -> A>>,
    last_render_time: Option<web_time::Instant>,
    #[cfg(target_arch = "wasm32")]
    proxy: EventLoopProxy<EngineEvent<A>>,

    fps_timer: f32,
    frame_count: u32,
}

impl<A: Application + 'static> ApplicationHandler<EngineEvent<A>> for EngineRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.is_none() {
            let mut attributes = Window::default_attributes();
            attributes.title = "Compute Engine".to_string();

            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;

                let window = web_sys::window().unwrap();
                let documents = window.document().unwrap();
                let canvas = documents.get_element_by_id("canvas").unwrap();
                let html_canvas_element: web_sys::HtmlCanvasElement = canvas.unchecked_into();

                html_canvas_element.set_tab_index(0);
                let _ = html_canvas_element.focus();

                attributes = attributes.with_canvas(Some(html_canvas_element));
            }

            let window = Arc::new(event_loop.create_window(attributes).unwrap());
            let builder = self.app_builder.take().expect("App builder Missing!");

            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut engine = pollster::block_on(Engine::new(window)).unwrap();
                let app = builder(&mut engine);

                self.engine = Some(engine);
                self.app = Some(app);
                self.last_render_time = Some(web_time::Instant::now());
            }

            #[cfg(target_arch = "wasm32")]
            {
                let proxy = self.proxy.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let mut engine = Engine::new(window)
                        .await
                        .expect("Failed to create engine!! Embarassing...");
                    let mut app = builder(&mut engine);

                    let size = engine.window.inner_size();
                    engine.resize(size.width, size.height);
                    app.resize(&mut engine, size.width, size.height);

                    proxy.send_event(EngineEvent::Initialized(engine, app)).ok();
                });
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: EngineEvent<A>) {
        let EngineEvent::Initialized(engine, app) = event;

        self.engine = Some(engine);
        self.app = Some(app);
        self.last_render_time = Some(web_time::Instant::now());

        if let Some(e) = self.engine.as_mut() {
            e.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let (engine, app) = match (self.engine.as_mut(), self.app.as_mut()) {
            (Some(e), Some(a)) => (e, a),
            _ => return,
        };

        if engine.input.process_event(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
                app.resize(engine, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                // Calculate Delta Time
                let now = web_time::Instant::now();
                let dt = match self.last_render_time {
                    Some(last) => now.duration_since(last).as_secs_f32(),
                    None => 0.0,
                };
                self.last_render_time = Some(now);

                // Run the application loop
                engine.update(dt);
                app.update(engine, dt);
                match app.render(engine) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Render Error: {:?}", e);
                    }
                }

                self.fps_timer += dt;
                self.frame_count += 1;

                if self.fps_timer >= 0.5 {
                    let fps = self.frame_count as f32 / self.fps_timer;

                    println!("Engine FPS: {:.1}", fps);

                    self.frame_count = 0;
                    self.fps_timer = 0.0;
                }

                engine.window.request_redraw();
                engine.input.finish_frame();
            }
            _ => {}
        }
    }
}

pub fn run<A: Application + 'static>(
    app_builder: impl FnOnce(&mut Engine) -> A + 'static,
) -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    #[cfg(target_arch = "wasm32")]
    console_log::init_with_level(log::Level::Info).unwrap_or(());

    let event_loop = EventLoop::<EngineEvent<A>>::with_user_event().build()?;

    let mut runner = EngineRunner {
        engine: None,
        app: None,
        app_builder: Some(Box::new(app_builder)),
        last_render_time: None,
        #[cfg(target_arch = "wasm32")]
        proxy: event_loop.create_proxy(),
        fps_timer: 0.0,
        frame_count: 0,
    };

    event_loop.run_app(&mut runner)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::prelude::JsValue> {
    console_error_panic_hook::set_once();

    crate::run(|engine| crate::app::raymarcher::RayMarcher::new(engine))
        .map_err(|e| wasm_bindgen::prelude::JsValue::from_str(&e.to_string()))?;

    Ok(())
}
