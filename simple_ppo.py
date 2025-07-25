import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class HumanoidEnv:
    """MuJoCo Humanoid Environment for Walking Training"""
    
    def __init__(self, xml_path, render_mode=None):
        self.xml_path = xml_path
        self.render_mode = render_mode
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize minimal viewer if rendering
        self.viewer = None
        if render_mode == "human":
            # Use passive viewer with minimal settings
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data,
                show_left_ui=False,      # Hide left panel
                show_right_ui=False,     # Hide right panel  
            )
            # Configure camera for better viewing
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = 1  # Track humanoid body
            # self.viewer.cam.distance = 4.0
            # self.viewer.cam.elevation = -20
            # self.viewer.cam.azimuth = 90
        
        # Action and observation space
        self.action_dim = self.model.nu  # Number of actuators
        self.obs_dim = self._get_obs().shape[0]
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Environment parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        self.initial_height = None
        
    def reset(self):
        """Reset environment to initial state"""
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random perturbation to initial state
        self.data.qpos[:] += np.random.normal(0, 0.01, self.data.qpos.shape)
        self.data.qvel[:] = np.random.normal(0, 0.1, self.data.qvel.shape)
        
        # Forward step to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        self.initial_height = self.data.qpos[2]  # Z position (height)
        
        return self._get_obs()
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Clip and scale actions
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        self.current_step += 1
        
        # Render if requested
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
            # Slower rendering for better visibility during testing
            time.sleep(0.02)  # 20ms delay for smoother viewing
        
        info = {
            'height': self.data.qpos[2],
            'forward_velocity': self.data.qvel[0],
            'step': self.current_step
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get current observation"""
        # Position (excluding global x,y to make problem translation invariant)
        qpos = self.data.qpos[2:].copy()  # Start from z-position
        
        # Velocity
        qvel = self.data.qvel.copy()
        
        # Additional observations
        height = self.data.qpos[2]
        
        # Center of mass
        com_pos = self.data.subtree_com[0].copy()
        com_vel = self.data.cvel[0][:3].copy()  # Linear velocity only
        
        # Concatenate all observations
        obs = np.concatenate([
            qpos,           # Joint positions
            qvel,           # Joint velocities  
            [height],       # Height
            com_pos,        # Center of mass position
            com_vel,        # Center of mass velocity
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on walking performance"""
        # Forward velocity reward
        forward_vel = self.data.qvel[0]
        forward_reward = forward_vel
        
        # Height penalty (encourage staying upright)
        height = self.data.qpos[2]
        height_reward = -abs(height - self.initial_height) * 2
        
        # Stability reward (penalize excessive angular velocity)
        angular_vel_penalty = -np.sum(np.square(self.data.qvel[3:6])) * 0.1
        
        # Energy efficiency (penalize large control inputs)
        control_penalty = -np.sum(np.square(self.data.ctrl)) * 0.01
        
        # Alive bonus
        alive_bonus = 1.0
        
        total_reward = (forward_reward + height_reward + 
                       angular_vel_penalty + control_penalty + alive_bonus)
        
        return total_reward
    
    def _is_done(self):
        """Check if episode should terminate"""
        height = self.data.qpos[2]
        
        # Terminate if robot falls (lowered threshold for testing)
        if height < 0.6:  # More lenient fall detection
            print(f"Robot fell! Height: {height:.3f}")
            return True
            
        # Terminate if max steps reached
        if self.current_step >= self.max_episode_steps:
            print("Max steps reached!")
            return True
            
        return False
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()


class PolicyNetwork(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Action standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Actor output
        action_mean = self.actor(shared_features)
        action_std = torch.exp(self.log_std)
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions):
        action_mean, action_std, values = self.forward(states)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(), entropy


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, 
                 lambda_gae=0.95, clip_eps=0.2, value_coef=0.5, 
                 entropy_coef=0.01, max_grad_norm=0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage
        self.reset_storage()
        
    def reset_storage(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] + self.gamma * values[t + 1] * 
                    (1 - self.dones[t]) - values[t])
            advantage = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * advantage
            advantages.insert(0, advantage)
            
        return advantages
    
    def update(self, next_value, epochs=10, batch_size=64):
        """Update policy using PPO"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute advantages and returns
        advantages = self.compute_gae(next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = len(self.states)
        
        for _ in range(epochs):
            # Create random batches
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions)
                
                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 
                                  1 + self.clip_eps) * batch_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + self.value_coef * value_loss + 
                            self.entropy_coef * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                             self.max_grad_norm)
                self.optimizer.step()
        
        # Clear storage
        self.reset_storage()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        
        return (action.cpu().numpy()[0], 
                log_prob.cpu().numpy()[0], 
                value.cpu().numpy()[0])
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        
    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))


def train_humanoid_walker(xml_path, total_timesteps=1000000, save_interval=10000):
    """Main training function"""
    
    # Create environment
    env = HumanoidEnv(xml_path, render_mode=None)  # Set to "human" for visualization
    
    # Create agent
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_eps=0.2
    )
    
    # Training variables
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    timestep = 0
    episode = 0
    
    print(f"Starting training...")
    print(f"Observation dim: {env.obs_dim}")
    print(f"Action dim: {env.action_dim}")
    
    while timestep < total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(env.max_episode_steps):
            # Get action from agent
            action, log_prob, value = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            state = next_state
            
            # Update policy every 2048 steps
            if len(agent.states) >= 2048:
                # Get final value for GAE computation
                if done:
                    next_value = 0
                else:
                    _, _, next_value = agent.get_action(next_state)
                
                # Update policy
                losses = agent.update(next_value)
                
                # Print progress
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                    avg_length = np.mean(episode_lengths) if episode_lengths else 0
                    print(f"Episode {episode}, Timestep {timestep}")
                    print(f"  Avg Reward: {avg_reward:.2f}")
                    print(f"  Avg Length: {avg_length:.1f}")
                    print(f"  Policy Loss: {losses['policy_loss']:.4f}")
                    print(f"  Value Loss: {losses['value_loss']:.4f}")
                    print("="*50)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1
        
        # Save model periodically
        if timestep % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/ppo_humanoid_{timestep}.pth")
            print(f"Model saved at timestep {timestep}")
    
    # Final save
    agent.save("models/ppo_humanoid_final.pth")
    env.close()
    
    return agent, episode_rewards


def test_trained_model(xml_path, model_path, episodes=5):
    """Test the trained model"""
    
    # Create environment with rendering
    env = HumanoidEnv(xml_path, render_mode="human")
    
    # Create and load agent
    agent = PPOAgent(env.obs_dim, env.action_dim)
    agent.load(model_path)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        print(f"Testing Episode {episode + 1}")
        
        for step in range(env.max_episode_steps):
            action, _, _ = agent.get_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}")
    
    env.close()

def test_sim():
    env = HumanoidEnv(model_path, render_mode="human")
    print(f"Action dimension: {env.action_dim}")
    print(f"Observation dimension: {env.obs_dim}")
    
    # Test 1: See robot in initial pose (no actions)
    print("\n=== Test 1: Initial Pose (5 seconds) ===")
    state = env.reset()
    print(f"Initial height: {state[0]:.3f}")  # Assuming height is first in observation
    
    for i in range(250):  # 5 seconds at 50Hz
        # Zero actions (no movement)
        zero_action = np.zeros(env.action_dim)
        state, reward, done, info = env.step(zero_action)
        
        if i % 50 == 0:
            print(f"Step {i}: Height = {info['height']:.3f}, Reward = {reward:.3f}")
        
        if done:
            print("Episode ended during initial pose test")
            break
    
    # Test 2: Small random movements
    print("\n=== Test 2: Small Random Movements ===")
    state = env.reset()
    
    for i in range(500):
        # Very small random actions to avoid instability
        action = np.random.uniform(-0.2, 0.2, env.action_dim)
        state, reward, done, info = env.step(action)
        
        if i % 50 == 0:
            print(f"Step {i}: Height = {info['height']:.3f}, Forward Vel = {info.get('forward_velocity', 0):.3f}")
        
        if done:
            print(f"Episode ended at step {i}")
            print(f"Final height: {info['height']:.3f}")
            break
    
    # Test 3: Check if the viewer window is responding
    print("\n=== Test 3: Viewer Response Test ===")
    print("The simulation window should be visible and updating.")
    print("Try moving/resizing the window to confirm it's responsive.")
    
    # Keep simulation running for manual inspection
    input("Press Enter to continue...")
    
    env.close()
    print("Simulation closed.")

if __name__ == "__main__":
    # Configuration
    model_path = os.path.join(SCRIPT_DIR, "g1/scene_23dof.xml")
    
    print("Testing Humanoid Simulation - Enhanced")
    
    
    
    # Test trained model
    test_trained_model(model_path, "models/ppo_humanoid_27.pth")
    
    # Plot training progress
    # plt.figure(figsize=(10, 6))
    # plt.plot(rewards)
    # plt.title('Training Progress')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()