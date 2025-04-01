# training/pg_training.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
sys.path.append("../")  # Add parent directory to path
from environment.custom_env import PatientMonitoringEnv

def train_ppo(timesteps=100000, log_interval=10, evaluation_episodes=10):
    """
    Train a PPO agent on the Patient Monitoring environment.
    
    Args:
        timesteps: Total timesteps for training
        log_interval: How often to log training progress
        evaluation_episodes: Number of episodes for final evaluation
    
    Returns:
        Trained model and evaluation results
    """
    # Create logs and model directories
    log_dir = "logs/ppo"
    model_dir = "models/pg"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize environment
    env = PatientMonitoringEnv()
    env = Monitor(env, log_dir)
    
    # Create evaluation environment
    eval_env = PatientMonitoringEnv()
    
    # Define model hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,  # Discount factor
        n_steps=2048,  # Steps to collect before updating
        batch_size=64,
        ent_coef=0.01,  # Entropy coefficient to encourage exploration
        clip_range=0.2,  # Clipping parameter for PPO
        n_epochs=10,  # Number of epochs when optimizing the surrogate loss
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log=log_dir,
        verbose=1
    )
    
    # Create a callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=model_dir,
        name_prefix="ppo_checkpoint"
    )
    
    # Train the model
    print(f"Training PPO for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        log_interval=log_interval,
        callback=checkpoint_callback
    )
    
    # Save final model
    model_path = f"{model_dir}/ppo_final_model"
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
    plt.title("PPO Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(episode_lengths)), episode_lengths)
    plt.title("PPO Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/ppo_performance.png")
    
    return model, (mean_reward, std_reward)

if __name__ == "__main__":
    train_ppo()