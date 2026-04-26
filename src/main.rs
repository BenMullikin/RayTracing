use raymarcher::app::raymarcher::RayMarcher;
use raymarcher::engine::core::Engine;

fn main() {
    raymarcher::run(|engine: &mut Engine| RayMarcher::new(engine)).unwrap();
}
