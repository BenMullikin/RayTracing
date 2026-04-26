use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct Input {
    keys_pressed: HashSet<KeyCode>,
    mouse_pressed: HashSet<MouseButton>,
    pub cursor_position: [f32; 2],
    pub cursor_delta: [f32; 2],
    pub mouse_scroll: f32,
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys_pressed: HashSet::new(),
            mouse_pressed: HashSet::new(),
            cursor_position: [0.0, 0.0],
            cursor_delta: [0.0, 0.0],
            mouse_scroll: 0.0,
        }
    }

    pub fn process_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                if *state == ElementState::Pressed {
                    self.keys_pressed.insert(*code);
                } else {
                    self.keys_pressed.remove(code);
                }
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    self.mouse_pressed.insert(*button);
                } else {
                    self.mouse_pressed.remove(button);
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = [position.x as f32, position.y as f32];
                // Calculate how far the mouse moved since the last event
                self.cursor_delta[0] += new_pos[0] - self.cursor_position[0];
                self.cursor_delta[1] += new_pos[1] - self.cursor_position[1];
                self.cursor_position = new_pos;
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.mouse_scroll = match delta {
                    MouseScrollDelta::LineDelta(_, scroll) => *scroll * 100.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                true
            }
            _ => false,
        }
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_pressed.contains(&button)
    }

    pub fn finish_frame(&mut self) {
        self.cursor_delta = [0.0, 0.0];
        self.mouse_scroll = 0.0;
    }
}
