use crate::engine::core::Engine;

pub mod raymarcher;

pub trait Application {
    fn update(&mut self, engine: &mut Engine, dt: f32);
    fn render(&mut self, engine: &mut Engine) -> anyhow::Result<()>;
    fn resize(&mut self, engine: &mut Engine, width: u32, height: u32);
}
