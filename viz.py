import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from collections import deque, defaultdict
import time
import json
from datetime import datetime
import pandas as pd
from matplotlib.animation import FuncAnimation
import threading
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RLTrainingVisualizer:
    """Comprehensive visualization system for RL training monitoring"""
    
    def __init__(self, run_dir, experiment_name="PPO_Humanoid"):
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        
        # Create tensorboard log directory
        self.tb_log_dir = os.path.join(run_dir, "tensorboard_logs")
        os.makedirs(self.tb_log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        
        # Storage for metrics
        self.metrics = defaultdict(list)
        self.episode_data = []
        self.timestep_data = []
        
        # Live plotting setup
        self.live_fig = None
        self.live_axes = None
        self.live_plots_enabled = False
        
        # Training metadata
        self.training_start_time = time.time()
        self.last_update_time = time.time()
        
        print(f"Visualization system initialized for: {experiment_name}")
        print(f"TensorBoard logs: {self.tb_log_dir}")
        print(f"Run 'tensorboard --logdir {self.tb_log_dir}' to view real-time plots")
    
    def log_episode_metrics(self, episode, timestep, episode_reward, episode_length, 
                          episode_info=None, reward_breakdown=None):
        """Log episode-level metrics"""
        
        # Store episode data
        episode_data = {
            'episode': episode,
            'timestep': timestep,
            'reward': episode_reward,
            'length': episode_length,
            'time': time.time()
        }
        
        if episode_info:
            episode_data.update(episode_info)
            
        if reward_breakdown:
            episode_data.update({f'reward_{k}': v for k, v in reward_breakdown.items()})
            
        self.episode_data.append(episode_data)
        
        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar('Episode/Timestep', timestep, episode)
        
        # Log reward breakdown if available
        if reward_breakdown:
            for reward_type, value in reward_breakdown.items():
                if reward_type != 'total':
                    self.writer.add_scalar(f'Rewards/{reward_type}', value, episode)
        
        # Log additional info
        if episode_info:
            for key, value in episode_info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Episode_Info/{key}', value, episode)
        
        # Update running statistics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['timesteps'].append(timestep)
        
        # Keep only last 1000 episodes for memory efficiency
        if len(self.metrics['episode_rewards']) > 1000:
            for key in self.metrics:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def log_training_metrics(self, timestep, losses, learning_stats=None):
        """Log training/optimization metrics"""
        
        # TensorBoard logging for losses
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(f'Training/{loss_name}', loss_value, timestep)
        
        # Log learning statistics
        if learning_stats:
            for stat_name, stat_value in learning_stats.items():
                if isinstance(stat_value, (int, float)):
                    self.writer.add_scalar(f'Learning/{stat_name}', stat_value, timestep)
        
        # Store for later analysis
        training_data = {
            'timestep': timestep,
            'time': time.time(),
            **losses
        }
        if learning_stats:
            training_data.update(learning_stats)
            
        self.timestep_data.append(training_data)
    
    def log_policy_metrics(self, timestep, policy_stats):
        """Log policy-specific metrics"""
        
        for stat_name, stat_value in policy_stats.items():
            if isinstance(stat_value, (int, float)):
                self.writer.add_scalar(f'Policy/{stat_name}', stat_value, timestep)
            elif isinstance(stat_value, torch.Tensor):
                if stat_value.numel() == 1:
                    self.writer.add_scalar(f'Policy/{stat_name}', stat_value.item(), timestep)
                else:
                    # For multi-dimensional tensors, log histogram
                    self.writer.add_histogram(f'Policy/{stat_name}', stat_value, timestep)
    
    def log_environment_metrics(self, timestep, env_stats):
        """Log environment-specific metrics"""
        
        for stat_name, stat_value in env_stats.items():
            if isinstance(stat_value, (int, float)):
                self.writer.add_scalar(f'Environment/{stat_name}', stat_value, timestep)
    
    def log_hyperparameters(self, hparams):
        """Log hyperparameters"""
        
        # Convert all values to strings for TensorBoard
        hparams_str = {k: str(v) for k, v in hparams.items()}
        self.writer.add_hparams(hparams_str, {})
        
        # Save to JSON for later reference
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
    
    def create_training_summary_plots(self, save_path=None):
        """Create comprehensive training summary plots"""
        
        if not self.episode_data:
            print("No episode data available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'{self.experiment_name} - Training Summary', fontsize=16)
        
        # Convert episode data to DataFrame for easier plotting
        df = pd.DataFrame(self.episode_data)
        
        # 1. Episode Rewards over Time
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
        if len(df) > 10:
            rolling_mean = df['reward'].rolling(window=min(50, len(df)//10)).mean()
            axes[0, 0].plot(df['episode'], rolling_mean, linewidth=2, label='Rolling Mean')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Episode Lengths
        axes[0, 1].plot(df['episode'], df['length'], alpha=0.3, label='Raw')
        if len(df) > 10:
            rolling_mean = df['length'].rolling(window=min(50, len(df)//10)).mean()
            axes[0, 1].plot(df['episode'], rolling_mean, linewidth=2, label='Rolling Mean')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Reward Distribution
        axes[0, 2].hist(df['reward'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(df['reward'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["reward"].mean():.2f}')
        axes[0, 2].set_title('Reward Distribution')
        axes[0, 2].set_xlabel('Reward')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training Progress (Sample Efficiency)
        axes[1, 0].plot(df['timestep'], df['reward'], alpha=0.6)
        axes[1, 0].set_title('Sample Efficiency')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Episode Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Reward Components (if available)
        reward_cols = [col for col in df.columns if col.startswith('reward_') and col != 'reward_total']
        if reward_cols:
            for col in reward_cols[:5]:  # Show max 5 components
                axes[1, 1].plot(df['episode'], df[col], label=col.replace('reward_', ''), alpha=0.7)
            axes[1, 1].set_title('Reward Components')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No reward breakdown\navailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Reward Components')
        
        # 6. Performance Statistics
        recent_rewards = df['reward'].tail(min(100, len(df)))
        stats_text = f"""
        Total Episodes: {len(df)}
        Total Timesteps: {df['timestep'].iloc[-1] if len(df) > 0 else 0}
        
        Recent Performance (last {len(recent_rewards)} episodes):
        Mean Reward: {recent_rewards.mean():.2f}
        Std Reward: {recent_rewards.std():.2f}
        Max Reward: {recent_rewards.max():.2f}
        Min Reward: {recent_rewards.min():.2f}
        
        Overall:
        Best Episode: {df['reward'].max():.2f}
        Worst Episode: {df['reward'].min():.2f}
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Training Statistics')
        axes[1, 2].axis('off')
        
        # 7. Learning Curve with Confidence Intervals
        if len(df) > 20:
            window_size = min(50, len(df) // 5)
            rolling_mean = df['reward'].rolling(window=window_size).mean()
            rolling_std = df['reward'].rolling(window=window_size).std()
            
            axes[2, 0].plot(df['episode'], rolling_mean, linewidth=2, label='Mean')
            axes[2, 0].fill_between(df['episode'], 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std, 
                                   alpha=0.3, label='±1 Std')
            axes[2, 0].set_title(f'Learning Curve (Window: {window_size})')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Reward')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].plot(df['episode'], df['reward'])
            axes[2, 0].set_title('Learning Curve')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Reward')
        
        # 8. Episode Length vs Reward Correlation
        axes[2, 1].scatter(df['length'], df['reward'], alpha=0.6)
        axes[2, 1].set_title('Episode Length vs Reward')
        axes[2, 1].set_xlabel('Episode Length')
        axes[2, 1].set_ylabel('Episode Reward')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(df) > 1:
            corr = df['length'].corr(df['reward'])
            axes[2, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[2, 1].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 9. Training Time Analysis
        if 'time' in df.columns:
            training_times = np.diff(df['time'])
            axes[2, 2].plot(df['episode'][1:], training_times, alpha=0.7)
            axes[2, 2].set_title('Episode Training Time')
            axes[2, 2].set_xlabel('Episode')
            axes[2, 2].set_ylabel('Time (seconds)')
            axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, 'No timing data\navailable', 
                           ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Training Time')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.run_dir, 'training_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training summary saved to: {save_path}")
        
        return fig
    
    def create_loss_analysis_plots(self, save_path=None):
        """Create detailed loss analysis plots"""
        
        if not self.timestep_data:
            print("No training data available for loss analysis")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.timestep_data)
        
        # Identify loss columns
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        
        if not loss_cols:
            print("No loss data found")
            return
        
        # Create subplots
        n_losses = len(loss_cols)
        n_cols = min(3, n_losses)
        n_rows = (n_losses + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{self.experiment_name} - Loss Analysis', fontsize=16)
        
        for i, loss_col in enumerate(loss_cols):
            if i < len(axes):
                axes[i].plot(df['timestep'], df[loss_col], alpha=0.7)
                
                # Add rolling average
                if len(df) > 10:
                    window = min(50, len(df) // 5)
                    rolling_mean = df[loss_col].rolling(window=window).mean()
                    axes[i].plot(df['timestep'], rolling_mean, linewidth=2, 
                               label=f'Rolling Mean (window={window})')
                
                axes[i].set_title(f'{loss_col.replace("_", " ").title()}')
                axes[i].set_xlabel('Timesteps')
                axes[i].set_ylabel('Loss Value')
                axes[i].grid(True, alpha=0.3)
                if len(df) > 10:
                    axes[i].legend()
        
        # Hide unused subplots
        for i in range(n_losses, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.run_dir, 'loss_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss analysis saved to: {save_path}")
        
        return fig
    
    def create_live_monitoring_dashboard(self):
        """Create a live updating dashboard (call in separate thread)"""
        
        plt.ion()  # Turn on interactive mode
        
        self.live_fig, self.live_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.live_fig.suptitle(f'{self.experiment_name} - Live Monitoring')
        
        self.live_plots_enabled = True
        
        def update_live_plots():
            while self.live_plots_enabled:
                try:
                    if self.episode_data:
                        df = pd.DataFrame(self.episode_data)
                        
                        # Clear axes
                        for ax in self.live_axes.flatten():
                            ax.clear()
                        
                        # Plot 1: Recent rewards
                        recent_episodes = df.tail(100)
                        self.live_axes[0, 0].plot(recent_episodes['episode'], recent_episodes['reward'])
                        self.live_axes[0, 0].set_title('Recent Episode Rewards')
                        self.live_axes[0, 0].set_xlabel('Episode')
                        self.live_axes[0, 0].set_ylabel('Reward')
                        self.live_axes[0, 0].grid(True, alpha=0.3)
                        
                        # Plot 2: Rolling average
                        if len(df) > 10:
                            window = min(20, len(df) // 2)
                            rolling_mean = df['reward'].rolling(window=window).mean()
                            self.live_axes[0, 1].plot(df['episode'], rolling_mean)
                            self.live_axes[0, 1].set_title(f'Rolling Average (window={window})')
                            self.live_axes[0, 1].set_xlabel('Episode')
                            self.live_axes[0, 1].set_ylabel('Average Reward')
                            self.live_axes[0, 1].grid(True, alpha=0.3)
                        
                        # Plot 3: Episode lengths
                        recent_lengths = df.tail(100)
                        self.live_axes[1, 0].plot(recent_lengths['episode'], recent_lengths['length'])
                        self.live_axes[1, 0].set_title('Recent Episode Lengths')
                        self.live_axes[1, 0].set_xlabel('Episode')
                        self.live_axes[1, 0].set_ylabel('Steps')
                        self.live_axes[1, 0].grid(True, alpha=0.3)
                        
                        # Plot 4: Current statistics
                        recent_rewards = df['reward'].tail(20)
                        stats_text = f"""
                        Episodes: {len(df)}
                        Last 20 episodes:
                        Mean: {recent_rewards.mean():.2f}
                        Std: {recent_rewards.std():.2f}
                        Max: {recent_rewards.max():.2f}
                        Min: {recent_rewards.min():.2f}
                        """
                        self.live_axes[1, 1].text(0.1, 0.5, stats_text, 
                                                 transform=self.live_axes[1, 1].transAxes,
                                                 fontfamily='monospace')
                        self.live_axes[1, 1].set_title('Current Statistics')
                        self.live_axes[1, 1].axis('off')
                        
                        plt.tight_layout()
                        plt.draw()
                        plt.pause(0.1)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    print(f"Live plotting error: {e}")
                    time.sleep(1)
        
        # Start live plotting in separate thread
        live_thread = threading.Thread(target=update_live_plots, daemon=True)
        live_thread.start()
        
        return self.live_fig
    
    def stop_live_monitoring(self):
        """Stop live monitoring dashboard"""
        self.live_plots_enabled = False
        if self.live_fig:
            plt.close(self.live_fig)
    
    def save_training_data(self):
        """Save all training data to files"""
        
        # Save episode data
        if self.episode_data:
            episode_df = pd.DataFrame(self.episode_data)
            episode_df.to_csv(os.path.join(self.run_dir, 'episode_data.csv'), index=False)
        
        # Save timestep data
        if self.timestep_data:
            timestep_df = pd.DataFrame(self.timestep_data)
            timestep_df.to_csv(os.path.join(self.run_dir, 'training_data.csv'), index=False)
        
        # Save summary statistics
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            summary = {
                'total_episodes': len(df),
                'total_timesteps': df['timestep'].iloc[-1] if len(df) > 0 else 0,
                'training_duration': time.time() - self.training_start_time,
                'final_performance': {
                    'last_100_episodes_mean': df['reward'].tail(100).mean(),
                    'last_100_episodes_std': df['reward'].tail(100).std(),
                    'best_episode_reward': df['reward'].max(),
                    'worst_episode_reward': df['reward'].min(),
                },
                'overall_performance': {
                    'mean_reward': df['reward'].mean(),
                    'std_reward': df['reward'].std(),
                    'mean_episode_length': df['length'].mean(),
                    'std_episode_length': df['length'].std(),
                }
            }
            
            with open(os.path.join(self.run_dir, 'training_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2, default=str)
    
    def close(self):
        """Clean up resources"""
        self.stop_live_monitoring()
        self.save_training_data()
        
        # Create final summary plots
        try:
            self.create_training_summary_plots()
            self.create_loss_analysis_plots()
        except Exception as e:
            print(f"Error creating final plots: {e}")
        
        # Close tensorboard writer
        self.writer.close()
        
        print(f"Visualization system closed. Final data saved to: {self.run_dir}")


def create_comparison_plots(run_dirs, experiment_names=None, save_path="comparison_plots.png"):
    """Compare multiple training runs"""
    
    if experiment_names is None:
        experiment_names = [f"Run {i+1}" for i in range(len(run_dirs))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Runs Comparison', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for i, (run_dir, name) in enumerate(zip(run_dirs, experiment_names)):
        episode_file = os.path.join(run_dir, 'episode_data.csv')
        
        if os.path.exists(episode_file):
            df = pd.read_csv(episode_file)
            
            # Plot 1: Episode rewards
            axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, color=colors[i])
            if len(df) > 20:
                rolling_mean = df['reward'].rolling(window=50).mean()
                axes[0, 0].plot(df['episode'], rolling_mean, 
                              linewidth=2, label=name, color=colors[i])
            else:
                axes[0, 0].plot(df['episode'], df['reward'], 
                              linewidth=2, label=name, color=colors[i])
            
            # Plot 2: Sample efficiency
            axes[0, 1].plot(df['timestep'], df['reward'], 
                          alpha=0.6, label=name, color=colors[i])
            
            # Plot 3: Episode lengths
            axes[1, 0].plot(df['episode'], df['length'], 
                          alpha=0.6, label=name, color=colors[i])
            
            # Plot 4: Final performance comparison (bar plot)
            final_performance = df['reward'].tail(50).mean()
            axes[1, 1].bar(i, final_performance, color=colors[i], alpha=0.7)
            axes[1, 1].set_xticks(range(len(experiment_names)))
            axes[1, 1].set_xticklabels(experiment_names, rotation=45)
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Sample Efficiency')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Final Performance (Last 50 Episodes)')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {save_path}")
    
    return fig


# Example usage and utility functions
def setup_visualization(run_dir, experiment_name="PPO_Humanoid", enable_live=False):
    """Setup visualization system for training"""
    
    visualizer = RLTrainingVisualizer(run_dir, experiment_name)
    
    if enable_live:
        visualizer.create_live_monitoring_dashboard()
    
    return visualizer


def analyze_training_run(run_dir):
    """Analyze a completed training run"""
    
    episode_file = os.path.join(run_dir, 'episode_data.csv')
    training_file = os.path.join(run_dir, 'training_data.csv')
    
    if not os.path.exists(episode_file):
        print(f"No episode data found in {run_dir}")
        return
    
    # Load data
    episode_df = pd.read_csv(episode_file)
    
    print(f"Training Analysis for: {run_dir}")
    print("="*50)
    print(f"Total Episodes: {len(episode_df)}")
    print(f"Total Timesteps: {episode_df['timestep'].iloc[-1]}")
    
    # Performance analysis
    final_100 = episode_df['reward'].tail(100)
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Mean Reward: {final_100.mean():.2f} ± {final_100.std():.2f}")
    print(f"  Best Episode: {final_100.max():.2f}")
    print(f"  Worst Episode: {final_100.min():.2f}")
    
    # Learning progress
    first_100 = episode_df['reward'].head(100)
    improvement = final_100.mean() - first_100.mean()
    print(f"\nLearning Progress:")
    print(f"  Initial Performance (first 100): {first_100.mean():.2f}")
    print(f"  Final Performance (last 100): {final_100.mean():.2f}")
    print(f"  Improvement: {improvement:.2f}")
    
    # Create analysis plots
    visualizer = RLTrainingVisualizer(run_dir, "Analysis")
    visualizer.episode_data = episode_df.to_dict('records')
    
    if os.path.exists(training_file):
        training_df = pd.read_csv(training_file)
        visualizer.timestep_data = training_df.to_dict('records')
    
    visualizer.create_training_summary_plots()
    visualizer.create_loss_analysis_plots()
    
    return episode_df