import pygame
import numpy as np
import imageio

# Medical Drone Theme Colors
colors = {
    'bg': (245, 245, 250),           # Light gray
    'grid': (200, 210, 230),         # Soft blue
    'patient_normal': (0, 180, 90),  # Green
    'patient_stroke': (230, 20, 20), # Red
    'drone': (0, 110, 240)           # Medical blue
}

def draw_grid(surface, grid_size, cell_size):
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, colors['grid'], rect, 2, border_radius=5)

def draw_patient(surface, patient_pos, stroke, cell_size):
    patient_color = colors['patient_stroke'] if stroke else colors['patient_normal']
    patient_pos_pix = ((patient_pos + 0.5) * cell_size).astype(int)
    pygame.draw.circle(surface, patient_color, patient_pos_pix, cell_size//3)

def draw_drone(surface, drone_pos, cell_size):
    drone_pos_pix = ((drone_pos + 0.5) * cell_size).astype(int)

    # Draw central body
    body_size = cell_size // 3
    pygame.draw.rect(surface, colors['drone'],
                     (*drone_pos_pix - body_size//2, body_size, body_size), border_radius=5)

    # Draw quadcopter rotors
    rotor_radius = cell_size // 8
    offset = body_size
    for dx, dy in [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]:
        rotor_pos = drone_pos_pix + np.array([dx, dy])
        pygame.draw.circle(surface, colors['drone'], rotor_pos, rotor_radius)

def visualize_environment(
    grid_size=10,
    cell_size=70,
    steps=100,
    drone_start=None,
    patient_start=None,
    filename='stroke_env_simulation.gif'
):
    window_size = grid_size * cell_size
    pygame.init()
    surface = pygame.Surface((window_size, window_size))
    frames = []

    # Initialize positions
    drone_pos = np.array(drone_start if drone_start is not None else [grid_size//2, grid_size//2])
    patient_pos = np.array(patient_start if patient_start is not None else [grid_size//2-1, grid_size//2+1])

    for _ in range(steps):
        stroke = np.random.rand() < 0.05
        # Random movements
        patient_pos += np.random.choice([-1, 0, 1], size=2)
        patient_pos = np.clip(patient_pos, 0, grid_size-1)

        drone_pos += np.random.choice([-1, 0, 1], size=2)
        drone_pos = np.clip(drone_pos, 0, grid_size-1)

        # Draw everything
        surface.fill(colors['bg'])
        draw_grid(surface, grid_size, cell_size)
        draw_patient(surface, patient_pos, stroke, cell_size)
        draw_drone(surface, drone_pos, cell_size)

        # Capture frame
        frame = pygame.surfarray.array3d(surface)
        frames.append(np.transpose(frame, (1, 0, 2)))

    # Save as GIF
    imageio.mimsave(filename, frames, duration=0.1)
    pygame.quit()
    print(f"GIF saved as '{filename}'")

if __name__ == "__main__":
    # Example usage
    visualize_environment(grid_size=10, cell_size=70, steps=100, filename='stroke_env_demo.gif')
