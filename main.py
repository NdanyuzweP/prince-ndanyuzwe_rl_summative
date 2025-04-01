#!/usr/bin/env python3
"""
Main script for running RL experiments on the Patient Monitoring environment.
This script allows for training, evaluating, and visualizing both DQN and PPO models.
"""
import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add the project root to Python path
import sys
sys.path.append(".")

# Import from local modules
from environment.custom_env import PatientMonitoringEnv
from environment.rendering import PatientMonitoringVisualization
from training.dqn_training import train_dqn
from training.pg_training import train_ppo

def main():
    """Main function to parse arguments and run experiments"""
    parser = argparse.ArgumentParser(description="RL Patient Monitoring System")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "visualize", "compare"], 
                      default="train", help="Run mode")
    parser.add_argument("--algorithm", type=str, choices=["dqn", "ppo", "both"], 
                      default="both", help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000, 
                      help="Number of timesteps for training")
    parser.add_argument("--render", action="store_true", 
                      help="Enable rendering during evaluation")
    parser.add_argument("--episodes", type=int, default=10, 
                      help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    
    # Training mode
    if args.mode == "train":
        results = {}
        
        if args.algorithm in ["dqn", "both"]:
            print("=== Training DQN Model ===")
            dqn_model, dqn_results = train_dqn(timesteps=args.timesteps)
            results["DQN"] = dqn_results
            
        if args.algorithm in ["ppo", "both"]:
            print("=== Training PPO Model ===")
            ppo_model, ppo_results = train_ppo(timesteps=args.timesteps)
            results["PPO"] = ppo_results
            
        if args.algorithm == "both":
            print("\n=== Training Results ===")
            for algo, (mean, std) in results.items():
                print(f"{algo}: Mean reward = {mean:.2f} ± {std:.2f}")
    
    # Evaluation mode        
    elif args.mode == "evaluate":
        # Load and evaluate models
        if args.algorithm in ["dqn", "both"]:
            try:
                dqn_model = DQN.load("models/dqn/dqn_final_model")
                eval_env = PatientMonitoringEnv(render_mode="human" if args.render else None)
                print("=== Evaluating DQN Model ===")
                
                mean_reward, std_reward = evaluate_policy(
                    dqn_model, eval_env, n_eval_episodes=args.episodes, deterministic=True
                )
                print(f"DQN: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
            except FileNotFoundError:
                print("DQN model not found. Train the model first.")
        
        if args.algorithm in ["ppo", "both"]:
            try:
                ppo_model = PPO.load("models/pg/ppo_final_model")
                eval_env = PatientMonitoringEnv(render_mode="human" if args.render else None)
                print("=== Evaluating PPO Model ===")
                
                mean_reward, std_reward = evaluate_policy(
                    ppo_model, eval_env, n_eval_episodes=args.episodes, deterministic=True
                )
                print(f"PPO: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
            except FileNotFoundError:
                print("PPO model not found. Train the model first.")
    
    # Visualization mode
    elif args.mode == "visualize":
        # Run the static OpenGL visualization
        print("Starting OpenGL Visualization of Patient Monitoring Environment")
        # Initialize with a patient in a mixed state for demonstration
        viz = PatientMonitoringVisualization(state=[0, 1, 2, 1])
        viz.run()
    
    # Compare mode to visualize training results
    elif args.mode == "compare":
        compare_algorithms()
        
def compare_algorithms():
    """Compare performance between DQN and PPO"""
    # Check if result files exist
    dqn_results_path = "logs/dqn/ppo_performance.png"
    ppo_results_path = "logs/ppo/ppo_performance.png"
    
    if not os.path.exists(dqn_results_path) or not os.path.exists(ppo_results_path):
        print("Training results not found. Run training first.")
        return
    
    try:
        # Load models for a direct comparison
        dqn_model = DQN.load("models/dqn/dqn_final_model")
        ppo_model = PPO.load("models/pg/ppo_final_model")
        
        # Create environments for evaluation
        env = PatientMonitoringEnv()
        
        # Evaluate both models on the same environment seeds
        num_episodes = 20
        dqn_rewards = []
        ppo_rewards = []
        
        for seed in range(num_episodes):
            env.reset(seed=seed)
            
            # Evaluate DQN
            obs, _ = env.reset(seed=seed)
            dqn_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = dqn_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                dqn_reward += reward
            
            dqn_rewards.append(dqn_reward)
            
            # Evaluate PPO
            obs, _ = env.reset(seed=seed)
            ppo_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ppo_reward += reward
            
            ppo_rewards.append(ppo_reward)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(num_episodes), dqn_rewards, alpha=0.5, label='DQN')
        plt.bar(range(num_episodes), ppo_rewards, alpha=0.5, label='PPO')
        plt.title('Episode Rewards Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        algorithms = ['DQN', 'PPO']
        avg_rewards = [np.mean(dqn_rewards), np.mean(ppo_rewards)]
        std_rewards = [np.std(dqn_rewards), np.std(ppo_rewards)]
        
        plt.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=10)
        plt.title('Average Performance Comparison')
        plt.ylabel('Mean Reward')
        
        plt.tight_layout()
        plt.savefig("logs/algorithm_comparison.png")
        plt.show()
        
        print(f"DQN Average Reward: {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
        print(f"PPO Average Reward: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
        
    except FileNotFoundError:
        print("Models not found. Train models first.")

if __name__ == "__main__":
    main()