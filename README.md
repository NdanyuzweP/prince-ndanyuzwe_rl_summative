# Patient Monitoring Reinforcement Learning System

This project implements a reinforcement learning-based patient monitoring system that learns to make appropriate decisions based on patient's vital signs. The system uses two different reinforcement learning algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), and compares their performance.

## Project Structure

```
project_root/
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
```

## Environment Description

The environment simulates a patient monitoring system where the agent must select appropriate interventions based on the patient's vital signs. 

### State Space
- Heart Rate (HR): Normal, Elevated, Critical
- Blood Pressure (BP): Normal, High, Very High
- Oxygen Saturation (SpO2): Normal, Low, Very Low
- Temperature (Temp): Normal, Fever, High Fever

### Action Space
- Action 0: Continue Monitoring
- Action 1: Send Mild Alert
- Action 2: Request Medical Evaluation
- Action 3: Activate Emergency Protocol

### Rewards
- Appropriate interventions receive positive rewards
- Inappropriate interventions receive negative rewards
- Successfully stabilizing the patient yields bonus rewards
- Failing to appropriately respond to critical situations incurs penalties

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/NdanyuzweP/prince-ndanyuzwe_rl_summative
cd patient_monitoring_rl
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Train both DQN and PPO models:
```bash
python main.py --mode train --algorithm both --timesteps 100000
```

Train only DQN model:
```bash
python main.py --mode train --algorithm dqn --timesteps 100000
```

Train only PPO model:
```bash
python main.py --mode train --algorithm ppo --timesteps 100000
```

### Evaluating Models

Evaluate both models:
```bash
python main.py --mode evaluate --algorithm both --episodes 10
```

Evaluate with rendering (to visualize agent behavior):
```bash
python main.py --mode evaluate --algorithm both --episodes 5 --render
```

### Visualization

Run the static OpenGL visualization:
```bash
python main.py --mode visualize
```

### Comparing Algorithms

Compare the performance of DQN and PPO:
```bash
python main.py --mode compare
```

## Results

After training, you can find:
1. Trained models in the `models/` directory
2. Training logs in the `logs/` directory
3. Performance plots showing episode rewards and lengths

The comparison mode generates a direct comparison between DQN and PPO, saved as `logs/algorithm_comparison.png`.

## Note

This project requires both Pygame and PyOpenGL for visualization. If you encounter any issues with the visualization, ensure these libraries are properly installed.
