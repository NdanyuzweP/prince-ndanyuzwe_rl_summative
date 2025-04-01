# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatientMonitoringEnv(gym.Env):
    """
    Custom Gym environment for proactive patient monitoring.
    The agent must select appropriate interventions based on patient vital signs.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super(PatientMonitoringEnv, self).__init__()
        
        # Define discrete state space (HR, BP, SpO2, Temp)
        self.state_space = {
            'HR': ['Normal', 'Elevated', 'Critical'],
            'BP': ['Normal', 'High', 'Very High'],
            'SpO2': ['Normal', 'Low', 'Very Low'],
            'Temp': ['Normal', 'Fever', 'High Fever']
        }
        
        # Define action space (Discrete: 4 actions)
        # 0: Continue Monitoring
        # 1: Send Mild Alert
        # 2: Request Medical Evaluation
        # 3: Activate Emergency Protocol
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (MultiDiscrete for categorical states)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.state = None
        self.time_step = 0
        self.max_time_steps = 50  # Limit per episode

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Start in a random state with some probability of critical values
        self.state = self.np_random.integers(0, 3, size=4)
        self.time_step = 0
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return np.array(self.state, dtype=np.int32), {}

    def step(self, action):
        """Execute agent action and transition to new state."""
        self.time_step += 1
        reward = 0
        terminated = False
        truncated = False

        # Extract current state values (HR, BP, SpO2, Temp)
        hr, bp, spo2, temp = self.state

        # Define action responses
        if action == 0:  # Continue Monitoring
            if all(val == 0 for val in self.state):  # All vitals normal
                reward = 2  # Appropriate monitoring
            else:
                reward = -2  # Should have acted on abnormal vitals
                
        elif action == 1:  # Send Mild Alert
            if hr == 1 or bp == 1 or spo2 == 1 or temp == 1:  # At least one elevated vital
                reward = 5  # Correct mild alert
            elif hr == 2 or bp == 2 or spo2 == 2 or temp == 2:  # Critical vital signs present
                reward = -3  # Should have escalated more
            else:
                reward = -5  # False alarm (all normal)
                
        elif action == 2:  # Request Medical Evaluation
            if hr == 2 or bp == 2 or spo2 == 2 or temp == 2:  # At least one critical vital
                reward = 10  # Correct critical alert
            elif hr == 1 or bp == 1 or spo2 == 1 or temp == 1:  # Only elevated but not critical
                reward = 3  # Slightly over-cautious but acceptable
            else:
                reward = -8  # Unnecessary escalation (all normal)
                
        elif action == 3:  # Activate Emergency Protocol
            critical_count = sum(1 for val in self.state if val == 2)
            if critical_count >= 2:  # Multiple critical vitals
                reward = 15  # Correct emergency response
            elif critical_count == 1:  # Only one critical vital
                reward = 5  # Somewhat justified but potentially overreaction
            else:
                reward = -10  # Completely unjustified emergency response
        
        # Simulate vital signs changing based on interventions and natural progression
        new_state = list(self.state)
        
        for i in range(4):
            # More intensive interventions have higher chance of improving vitals
            improvement_chance = 0.1 + (action * 0.1)
            deterioration_chance = 0.2 - (action * 0.05)
            
            # Random change based on probabilities
            change_val = self.np_random.choice(
                [-1, 0, 1], 
                p=[improvement_chance, 1-improvement_chance-deterioration_chance, deterioration_chance]
            )
            
            new_state[i] = max(0, min(2, new_state[i] + change_val))
        
        self.state = np.array(new_state, dtype=np.int32)
        
        # Terminal conditions
        if all(val == 0 for val in self.state):  # Patient fully stabilized
            terminated = True
            reward += 10
        elif sum(val == 2 for val in self.state) >= 3:  # Multiple critical values - patient critical
            if action < 2:  # Insufficient action for critical patient
                reward -= 15
        
        # Episode length limit
        if self.time_step >= self.max_time_steps:
            truncated = True
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return np.array(self.state, dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")
            
        # Initialize pygame if needed
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((600, 400))
            pygame.display.set_caption("Patient Monitoring System")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((600, 400))
        canvas.fill((0, 0, 0))
        
        # Define colors for different states
        colors = [
            [(0, 255, 0), (255, 255, 0), (255, 0, 0)],  # HR: Green, Yellow, Red
            [(0, 255, 0), (255, 255, 0), (255, 0, 0)],  # BP: Green, Yellow, Red
            [(0, 255, 0), (255, 255, 0), (255, 0, 0)],  # SpO2: Green, Yellow, Red
            [(0, 255, 0), (255, 255, 0), (255, 0, 0)],  # Temp: Green, Yellow, Red
        ]
        
        # Draw vital signs as colored circles
        labels = ["HR", "BP", "SpO2", "Temp"]
        for i, (vital, value) in enumerate(zip(labels, self.state)):
            x = 100 + i * 150
            color = colors[i][value]
            
            # Draw circle
            pygame.gfxdraw.filled_circle(canvas, x, 150, 30, color)
            pygame.gfxdraw.aacircle(canvas, x, 150, 30, (255, 255, 255))
            
            # Add labels
            if self.render_mode == "human":
                font = pygame.font.SysFont(None, 24)
                text = font.render(f"{vital}: {self.state_space[vital][value]}", True, (255, 255, 255))
                canvas.blit(text, (x - 40, 200))
                
                # Display time step
                step_text = font.render(f"Time Step: {self.time_step}", True, (255, 255, 255))
                canvas.blit(step_text, (20, 20))
            
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        """Close environment."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()