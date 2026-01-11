# safe-rl-gridworld
A reinforcement learning framework demonstrating safe exploration in grid environments where agents learn to clean while avoiding danger zones
safe-rl-gridworld
Safe Reinforcement Learning Gridworld provides a specialized environment for training agents to navigate a coordinate system while adhering to strict safety constraints. The project implements a risk-aware Q-learning agent that must balance the objective of cleaning messes with the high-stakes requirement of avoiding proximity to defined danger zones.

Environment Architecture
The core of this project is the SafeExplorationEnv class which simulates a discrete two dimensional world. Unlike standard gridworlds, this environment incorporates a safety buffer logic where any state within a specified Manhattan distance of a hazard is considered a terminal failure.

State Space
The agent perceives the world as a combined representation of its current coordinates and the remaining locations of messes. This ensures the environment remains Markovian while allowing the agent to learn optimal sequences for task completion.

Reward Mechanism
A sparse reward structure is utilized to guide the agent. Successful mess collection yields a positive reward of 10.0 while every movement incurs a minor penalty of -0.1 to discourage aimless wandering. Any violation of safety protocols results in a massive penalty of -20.0 and immediate episode termination.

Implementation Details
The learning process utilizes a Tabular Q-learning algorithm enhanced with exponential epsilon decay. This strategy allows for high initial exploration to discover the layout of danger zones followed by a transition to stable exploitation of the learned safety boundaries.

Core Dependencies
Numpy for numerical operations and coordinate clipping

Matplotlib for visualizing reward convergence over time

Tqdm for monitoring training progress across thousands of episodes

Getting Started
To execute the training simulation run the main script. The agent will initialize a five by five grid and begin learning through trial and error.

Bash

python safe_q_train.py
Upon completion the system generates a performance graph showing the moving average of rewards. A successful training run is characterized by a curve that starts low due to safety violations and climbs steadily as the agent learns to navigate around the danger buffers.

Customization
Users can modify the danger_zones set in the script to create different obstacle configurations. The safety_buffer parameter within the step function can also be adjusted to increase or decrease the strictness of the avoidance logic.
