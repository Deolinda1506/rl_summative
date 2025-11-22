import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Updated colors
colors = {
    'bg': (10, 10, 40),
    'grid': (55, 90, 140),
    'patient_normal': (0, 200, 120),
    'patient_stroke': (255, 80, 80),
    'drone': (255, 215, 0),  


class StrokeDetectionEnv(gym.Env):
    """Custom Gym environment for drone-based stroke detection."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        # Grid and agent settings
        self.grid_size = 10
        self.drone_pos = np.array([5, 5])
        self.patient_pos = np.random.randint(0, self.grid_size, size=2)
        self.stroke = False
        self.time_step = 0

        # Action space: 0-left, 1-right, 2-up, 3-down, 4-zoom_in, 5-zoom_out
        self.action_space = spaces.Discrete(6)

        # Observation: drone_x, drone_y, zoom, patient_x, patient_y
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(5,), dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self.window_size = 700
        self.cell_size = self.window_size // self.grid_size
        self.zoom = 1
        if render_mode == "rgb_array":
            pygame.init()
            self.window = pygame.Surface((self.window_size, self.window_size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([5, 5])
        self.patient_pos = np.random.randint(0, self.grid_size, size=2)
        self.stroke = False
        self.time_step = 0
        self.zoom = 1
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.drone_pos[0],
            self.drone_pos[1],
            self.zoom,
            self.patient_pos[0],
            self.patient_pos[1]
        ], dtype=np.float32)

    def step(self, action):
        reward = -0.1
        self.time_step += 1

        # Patient random movement
        self.patient_pos += np.random.choice([-1, 0, 1], size=2)
        self.patient_pos = np.clip(self.patient_pos, 0, self.grid_size-1)

        # Random stroke event (5% chance)
        self.stroke = np.random.rand() < 0.05

        # Drone actions
        if action == 0:
            self.drone_pos[0] -= 1
        elif action == 1:
            self.drone_pos[0] += 1
        elif action == 2:
            self.drone_pos[1] -= 1
        elif action == 3:
            self.drone_pos[1] += 1
        elif action == 4:
            self.zoom = min(self.zoom + 1, 5)
        elif action == 5:
            self.zoom = max(self.zoom - 1, 1)

        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size-1)

        # Reward for detecting stroke
        if self.stroke and np.array_equal(self.drone_pos, self.patient_pos):
            reward += 10
        elif self.stroke and not np.array_equal(self.drone_pos, self.patient_pos):
            reward -= 10
        elif not self.stroke and np.array_equal(self.drone_pos, self.patient_pos):
            reward -= 5

        done = self.time_step >= 200
        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Step:{self.time_step}, Drone:{self.drone_pos}, "
                  f"Patient:{self.patient_pos}, Stroke:{self.stroke}")
        elif self.render_mode == "rgb_array":
            self.window.fill(colors['bg'])

            # Draw grid
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                                       self.cell_size, self.cell_size)
                    pygame.draw.rect(self.window, colors['grid'], rect, 2, border_radius=8)

            # Draw patient
            patient_color = colors['patient_stroke'] if self.stroke else colors['patient_normal']
            patient_pix = ((self.patient_pos + 0.5) * self.cell_size).astype(int)
            pygame.draw.circle(self.window, patient_color, patient_pix, self.cell_size//3)

            # Draw drone
            drone_pix = ((self.drone_pos + 0.5) * self.cell_size).astype(int)
            pygame.draw.rect(self.window, colors['drone'],
                             (*drone_pix - self.cell_size//4, self.cell_size//2, self.cell_size//2), border_radius=10)

            for angle in [45, 135, 225, 315]:
                offset = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * self.cell_size//2.5
                rotor_pix = drone_pix + offset.astype(int)
                pygame.draw.circle(self.window, colors['drone'], rotor_pix, self.cell_size//10)

            return np.transpose(pygame.surfarray.array3d(self.window), (1, 0, 2))

    def close(self):
        pygame.quit()
