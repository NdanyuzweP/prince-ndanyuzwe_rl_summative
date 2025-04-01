<<<<<<< HEAD
Patient Monitoring Reinforcement Learning System
This project implements a reinforcement learning-based patient monitoring system that learns to make appropriate decisions based on patient's vital signs. The system uses two different reinforcement learning algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), and compares their performance.
Project Structure
Copyproject_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization components using PyOpenGL
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
Environment Description
The environment simulates a patient monitoring system where the agent must select appropriate interventions based on the patient's vital signs.
State Space

Heart Rate (HR): Normal, Elevated, Critical
Blood Pressure (BP): Normal, High, Very High
Oxygen Saturation (SpO2): Normal, Low, Very Low
Temperature (Temp): Normal, Fever, High Fever

Action Space

Action 0: Continue Monitoring
Action 1: Send Mild Alert
Action 2: Request Medical Evaluation
Action 3: Activate Emergency Protocol

Rewards

Appropriate interventions receive positive rewards
Inappropriate interventions receive negative rewards
Successfully stabilizing the patient yields bonus rewards
Failing to appropriately respond to critical situations incurs penalties

Installation and Setup

Clone the repository:

bashCopygit clone https://github.com/your-username/patient_monitoring_rl.git
cd patient_monitoring_rl

Create a virtual environment (recommended):

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt
Usage
Training Models
Train both DQN and PPO models:
bashCopypython main.py --mode train --algorithm both --timesteps 100000
Train only DQN model:
bashCopypython main.py --mode train --algorithm dqn --timesteps 100000
Train only PPO model:
bashCopypython main.py --mode train --algorithm ppo --timesteps 100000
Evaluating Models
Evaluate both models:
bashCopypython main.py --mode evaluate --algorithm both --episodes 10
Evaluate with rendering (to visualize agent behavior):
bashCopypython main.py --mode evaluate --algorithm both --episodes 5 --render
Visualization
Run the static OpenGL visualization:
bashCopypython main.py --mode visualize
Comparing Algorithms
Compare the performance of DQN and PPO:
bashCopypython main.py --mode compare
Results
After training, you can find:

Trained models in the models/ directory
Training logs in the logs/ directory
Performance plots showing episode rewards and lengths

The comparison mode generates a direct comparison between DQN and PPO, saved as 
=======
# Patient Monitoring Reinforcement Learning System

This project implements a reinforcement learning system for proactive patient monitoring in healthcare settings. The system trains agents to learn optimal intervention policies based on patient vital signs, helping to determine when medical attention is needed.

## Project Overview

The system uses a custom Gymnasium environment that simulates patient vital sign monitoring. Two reinforcement learning algorithms are implemented and compared:
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)

The agents learn to select appropriate interventions based on patient vital signs, balancing the need for timely intervention against the risk of false alarms.

## Environment

The environment simulates a patient monitoring system with:
- Four vital signs: Heart Rate (HR), Blood Pressure (BP), Oxygen Saturation (SpO2), and Temperature
- Four possible actions: Continue Monitoring, Send Mild Alert, Request Medical Evaluation, Activate Emergency Protocol

![Patient Monitoring System](docs/environment_screenshot.png)

## Features

- Custom Gymnasium environment for patient monitoring
- 3D visualization system using PyOpenGL
- Implementation of DQN and PPO algorithms using Stable-Baselines3
- Performance comparison between value-based and policy-based methods
- Checkpoint saving during training
- Evaluation and visualization tools

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   ├── rendering.py         # PyOpenGL visualization
├── training/
│   ├── dqn_training.py      # DQN training implementation
│   ├── pg_training.py       # PPO training implementation
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved PPO models
├── logs/
│   ├── dqn/                 # DQN training logs
│   └── ppo/                 # PPO training logs
├── main.py                  # Main script for running experiments
├── requirements.txt         # Project dependencies
└── README.md                # This documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/patient_monitoring_rl.git
cd patient_monitoring_rl
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train both DQN and PPO models:
```bash
python main.py --mode train --algorithm both --timesteps 100000
```

To train just one algorithm:
```bash
python main.py --mode train --algorithm dqn --timesteps 100000
```

### Evaluation

To evaluate the trained models:
```bash
python main.py --mode evaluate --algorithm both --episodes 10 --render
```

### Visualization

To run the 3D visualization:
```bash
python main.py --mode visualize
```

### Compare Algorithms

To compare the performance of DQN and PPO:
```bash
python main.py --mode compare
```

## Results

The implementation demonstrates how reinforcement learning can be applied to healthcare monitoring systems. Both DQN and PPO successfully learn effective intervention policies, with PPO generally showing more stable learning curves and better performance in terms of average rewards.

Detailed performance comparisons and visualizations can be found in the accompanying report.

## Dependencies

- gymnasium==0.28.1
- numpy==1.24.3
- stable-baselines3==2.1.0
- torch==2.0.1
- pygame==2.5.0
- PyOpenGL==3.1.6
- matplotlib==3.7.1
>>>>>>> 9314309f89690e9b432536c59ad78a78c763d7d3
