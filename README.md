
# DRL Welding – ABB Robot Collision‑Free Trajectory

This repository trains a deep reinforcement learning agent (PPO) to control an ABB welding robot
through the **Externally Guided Motion (EGM)** interface in RobotStudio.  
The agent learns to follow a weld seam while avoiding obstacles, keeping correct torch orientation,
and maintaining industrial process constraints – all from trial and error.

## Setup

1. Install Python 3.9+ and required packages:
   ```bash
   pip install -r requirements.txt
