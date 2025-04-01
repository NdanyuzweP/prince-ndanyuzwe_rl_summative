# training/dqn_training.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
sys.path.append("../")  # Add parent directory to path
from environment.custom_env import PatientMonitoringEnv

def train_dqn(timesteps=100000, log_interval=10, evaluation_episodes=10):
    """
    Train a DQN agent on the Patient Monitoring environment.
    
    Args:
        timesteps: Total timesteps for training
        log_interval: How often to log training progress
        evaluation_episodes: Number of episodes for final evaluation
    
    Returns:
        Trained model and evaluation results
    """
    # Create logs and model directories
    log_dir = "logs/dqn"
    model_dir = "models/dqn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize environment
    env = PatientMonitoringEnv()
    env = Monitor(env, log_dir)
    
    # Create evaluation environment
    eval_env = PatientMonitoringEnv()
    
    # Define model hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        gamma=0.99,  # Discount factor
        batch_size=64,
        buffer_size=10000,  # Replay buffer size
        exploration_fraction=0.1,  # Fraction of timesteps to explore
        exploration_final_eps=0.05,  # Final exploration rate
        target_update_interval=1000,  # How often to update target network
        train_freq=4,  # Update the model every 4 steps
        gradient_steps=1,  # How many gradient steps after each update
        tensorboard_log=log_dir,
        verbose=1
    )
    
    # Create a callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=model_dir,
        name_prefix="dqn_checkpoint"
    )
    
    # Train the model
    print(f"Training DQN for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        log_interval=log_interval,
        callback=checkpoint_callback
    )
    
    # Save final model
    model_path = f"{model_dir}/dqn_final_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate trained model
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=evaluation_episodes, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Record episode rewards for visualization
    rewards = []
    episode_lengths = []
    
    for _ in range(5):  # Run 5 episodes with the trained model
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Generate and save performance plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(rewards)), rewards)
    plt.title("DQN Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(episode_lengths)), episode_lengths)
    plt.title("DQN Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/dqn_performance.png")
    
    return model, (mean_reward, std_reward)

if __name__ == "__main__":
    train_dqn()