"""
Reward functions for humanoid walking training.
Based on common reward components used in locomotion tasks.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union


class BaseReward(ABC):
    """Base class for reward functions"""
    
    @abstractmethod
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        """Calculate reward given current state"""
        pass


class NaiveForwardReward(BaseReward):
    """Simple reward for moving forward in the X-direction."""
    
    def __init__(self, clip_min: Optional[float] = None, clip_max: Optional[float] = None, 
                 in_robot_frame: bool = True, weight: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.in_robot_frame = in_robot_frame
        self.weight = weight
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        # Linear velocity (first 3 components of qvel)
        linvel = qvel[:3]
        
        if self.in_robot_frame:
            # Transform velocity to robot frame using quaternion
            linvel = self._rotate_vector_by_quat(linvel, qpos[3:7], inverse=True)
        
        # Forward velocity (X-direction)
        forward_vel = linvel[0]
        
        # Apply clipping if specified
        if self.clip_min is not None:
            forward_vel = max(forward_vel, self.clip_min)
        if self.clip_max is not None:
            forward_vel = min(forward_vel, self.clip_max)
            
        return self.weight * forward_vel
    
    def _rotate_vector_by_quat(self, vector: np.ndarray, quat: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate vector by quaternion (simplified version)"""
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        if inverse:
            # Conjugate quaternion for inverse rotation
            quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        # Convert to rotation matrix and apply
        w, x, y, z = quat
        
        # Rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R @ vector


class NaiveForwardOrientationReward(NaiveForwardReward):
    """Simple reward for keeping the robot oriented in the X-direction."""
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        quat = qpos[3:7]
        forward_vec = np.array([1.0, 0.0, 0.0])
        
        # Rotate forward vector to current orientation
        forward_vec_rotated = self._rotate_vector_by_quat(forward_vec, quat, inverse=True)
        
        # Reward forward alignment, penalize lateral deviation
        forward_component = forward_vec_rotated[0]
        lateral_deviation = np.linalg.norm(forward_vec_rotated[1:])
        
        return self.weight * (forward_component - lateral_deviation)


class AngularVelocityReward(BaseReward):
    """Penalty for how fast the robot is rotating (stability reward)."""
    
    def __init__(self, axes: str = "xy", clip_min: Optional[float] = None, 
                 clip_max: Optional[float] = None, in_robot_frame: bool = True, 
                 weight: float = -0.1):
        self.axes = axes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.in_robot_frame = in_robot_frame
        self.weight = weight
        
        # Convert axes string to indices
        axis_map = {"x": 0, "y": 1, "z": 2}
        self.dims = [axis_map[ax] for ax in axes.lower()]
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        # Angular velocity (components 3:6 of qvel)
        angvel = qvel[3:6]
        
        if self.in_robot_frame:
            # Transform to robot frame
            angvel = self._rotate_vector_by_quat(angvel, qpos[3:7], inverse=True)
        
        # Select specified dimensions
        selected_angvel = angvel[self.dims]
        
        # Apply clipping
        if self.clip_min is not None or self.clip_max is not None:
            selected_angvel = np.clip(selected_angvel, self.clip_min, self.clip_max)
        
        # L2 norm of angular velocities (penalty for rotation)
        angular_penalty = np.linalg.norm(selected_angvel)
        
        return self.weight * angular_penalty
    
    def _rotate_vector_by_quat(self, vector: np.ndarray, quat: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Same rotation function as NaiveForwardReward"""
        quat = quat / np.linalg.norm(quat)
        
        if inverse:
            quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R @ vector


class StayAliveReward(BaseReward):
    """Reward for staying alive with penalty on termination."""
    
    def __init__(self, balance: float = 10.0, success_reward: Optional[float] = None, weight: float = 1.0):
        self.balance = balance
        self.success_reward = success_reward
        self.weight = weight
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        if done:
            if success:
                reward = 1.0 / self.balance if self.success_reward is None else self.success_reward
            else:
                reward = -1.0  # Penalty for failure/falling
        else:
            reward = 1.0 / self.balance  # Small positive reward for staying alive
            
        return self.weight * reward


class UprightReward(BaseReward):
    """Reward for staying upright (maintaining proper orientation)."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, success: bool = False) -> float:
        # Local up vector (z-axis)
        local_z = np.array([0.0, 0.0, 1.0])
        quat = qpos[3:7]
        
        # Transform local z to global frame
        global_z = self._rotate_vector_by_quat(local_z, quat)
        
        # Reward upright orientation, penalize tilting
        upright_component = global_z[2]  # Z component should be close to 1
        tilt_penalty = np.linalg.norm(global_z[:2])  # X,Y components should be close to 0
        
        return self.weight * (upright_component - tilt_penalty)
    
    def _rotate_vector_by_quat(self, vector: np.ndarray, quat: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Same rotation function"""
        quat = quat / np.linalg.norm(quat)
        
        if inverse:
            quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R @ vector


class EnergyEfficiencyReward(BaseReward):
    """Penalty for excessive control effort (energy efficiency)."""
    
    def __init__(self, weight: float = -0.01):
        self.weight = weight
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, 
                   success: bool = False, ctrl: Optional[np.ndarray] = None) -> float:
        if ctrl is None:
            return 0.0
        
        # L2 norm of control inputs (energy penalty)
        energy_penalty = np.sum(np.square(ctrl))
        
        return self.weight * energy_penalty


class CompositeReward:
    """Combines multiple reward components with weights."""
    
    def __init__(self, reward_components: list[BaseReward]):
        self.reward_components = reward_components
    
    def get_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, 
                   success: bool = False, ctrl: Optional[np.ndarray] = None) -> dict:
        """Get reward from all components"""
        rewards = {}
        total_reward = 0.0
        
        for i, component in enumerate(self.reward_components):
            component_name = component.__class__.__name__
            
            # Handle special case for energy reward
            if isinstance(component, EnergyEfficiencyReward):
                reward_value = component.get_reward(qpos, qvel, done, success, ctrl)
            else:
                reward_value = component.get_reward(qpos, qvel, done, success)
            
            rewards[component_name] = reward_value
            total_reward += reward_value
        
        rewards['total'] = total_reward
        return rewards
    
    def get_total_reward(self, qpos: np.ndarray, qvel: np.ndarray, done: bool = False, 
                        success: bool = False, ctrl: Optional[np.ndarray] = None) -> float:
        """Get only the total reward value"""
        return self.get_reward(qpos, qvel, done, success, ctrl)['total']