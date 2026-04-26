use crate::engine::input::Input;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

pub struct Camera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub sensitivity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub position: [f32; 3],
    _padding1: u32,
    pub forward: [f32; 3],
    _padding2: u32,
    pub right: [f32; 3],
    _padding3: u32,
    pub up: [f32; 3],
    _padding4: u32,
}

impl Camera {
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            position,
            yaw: 0.0, //-90.0_f32.to_radians(),
            pitch: 0.0,
            speed: 5.0,
            sensitivity: 0.005,
        }
    }

    pub fn update(&mut self, input: &Input, dt: f32) {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let forward = [yaw_cos, 0.0, yaw_sin];
        let right = [-yaw_sin, 0.0, yaw_cos];

        let mut velocity = self.speed * dt;
        if input.is_key_pressed(KeyCode::ControlLeft) {
            velocity *= 3.0;
        }

        if input.is_key_pressed(KeyCode::KeyW) {
            self.position[0] += forward[0] * velocity;
            self.position[1] += forward[1] * velocity;
            self.position[2] += forward[2] * velocity;
        }
        if input.is_key_pressed(KeyCode::KeyS) {
            self.position[0] -= forward[0] * velocity;
            self.position[1] -= forward[1] * velocity;
            self.position[2] -= forward[2] * velocity;
        }
        if input.is_key_pressed(KeyCode::KeyD) {
            self.position[0] += right[0] * velocity;
            self.position[2] += right[2] * velocity;
        }
        if input.is_key_pressed(KeyCode::KeyA) {
            self.position[0] -= right[0] * velocity;
            self.position[2] -= right[2] * velocity;
        }
        if input.is_key_pressed(KeyCode::Space) {
            self.position[1] += velocity;
        }
        if input.is_key_pressed(KeyCode::ShiftLeft) {
            self.position[1] -= velocity;
        }

        if input.is_mouse_pressed(MouseButton::Left) {
            self.yaw += input.cursor_delta[0] * self.sensitivity;
            self.pitch -= input.cursor_delta[1] * self.sensitivity;

            let safe_angle = std::f32::consts::FRAC_PI_2 - 0.01;
            self.pitch = self.pitch.clamp(-safe_angle, safe_angle);
        }
    }

    pub fn get_uniform(&self) -> CameraUniform {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();

        let forward = [yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos];

        let right = [-forward[2], 0.0, forward[0]];

        let right_len = (right[0] * right[0] + right[2] * right[2]).sqrt();
        let right_norm = [right[0] / right_len, 0.0, right[2] / right_len];

        let up = [
            right_norm[1] * forward[2] - right_norm[2] * forward[1],
            right_norm[2] * forward[0] - right_norm[0] * forward[2],
            right_norm[0] * forward[1] - right_norm[1] * forward[0],
        ];
        CameraUniform {
            position: self.position,
            _padding1: 0,
            forward,
            _padding2: 0,
            right: right_norm,
            _padding3: 0,
            up,
            _padding4: 0,
        }
    }
}
