# ZENITH: Zero-shot Exploration and Navigation in Intelligent Textual Horizons

## Overview

**ZENITH** is a research project that explores how Large Language Models (LLMs) can perform zero-shot, waypoint-based navigation in simulated 3D urban environments. Instead of relying on precomputed maps or explicit training, ZENITH leverages real-time visual observations and prompt-based reasoning to generate and select intermediate navigation goals (waypoints). The project uses the **SimWorld** platform (Unreal Engine 5) for high-fidelity, city-scale simulations.

---

## Features

- **Zero-shot waypoint planning:** No environment-specific training required; navigation is driven by LLM reasoning over real-time multimodal inputs.
- **Multimodal input:** Agents receive RGB images, depth maps, and segmentation masks at each step.
- **Waypoint-based navigation:** LLMs propose and select waypoints as intermediate goals using visual and spatial cues.
- **Closed-loop control:** The agent autonomously interfaces with the LLM, executing navigation actions based on model outputs.
- **Evaluation metrics:** Success rates for both waypoint selection and final goal-reaching are reported, with ablation studies comparing LLM reasoning to hardcoded controllers.

---

## Motivation

Traditional navigation agents often rely on static maps or require extensive training. In dynamic or partially observable environments, agents must reason over both high-level goals and local observations. ZENITH investigates whether LLMs can serve as flexible reasoning modules for navigation, particularly in settings where only real-time visual data is available and the environment may change unpredictably.

---

## Methodology

### Environment

- **SimWorld (UE5):** Procedurally generates realistic urban scenes with buildings, roads, and dynamic obstacles.
- **Agent Observations:** At each timestep, the agent receives:
  - RGB image (first-person view)
  - Depth map
  - Segmentation mask (ground vs. obstacles)
  - Current pose and target coordinates

### Waypoint Generation and Selection

1. **Generation:** The LLM (e.g., GPT-4o-mini) is prompted with the current visual data to propose 10–12 candidate waypoints, ensuring they are on navigable ground and widely distributed.
2. **Conversion:** Candidate waypoints (image-plane coordinates) are mapped to 3D world coordinates using camera intrinsics and depth data.
3. **Selection:** The LLM selects the optimal waypoint based on criteria such as proximity to the target, path clarity, and avoidance of previously visited or obstructed areas.
4. **Execution:** The agent moves toward the selected waypoint, repeating the process until the goal is reached or a stopping condition is met.

---

## Experiments

- **Setup:** Agents start at random positions in city-scale scenes and must reach randomly assigned targets using only visual inputs.
- **Metrics:**
  - **Waypoint Selection Success Rate:** Proportion of steps where the LLM selects the optimal waypoint (closest valid point to the goal).
  - **Goal Success Rate:** Proportion of episodes where the agent reaches within 50 units of the target without collision or looping.

| Model         | Waypoint SR | Goal SR |
|---------------|-------------|---------|
| GPT-4o mini   | 77%         | 12.5%   |

- **Ablation Study:** Compared hardcoded controllers, standard LLM prompting, and Chain-of-Thought (CoT) prompting for local movement execution. Hardcoded controllers were most reliable, but CoT prompting improved LLM reasoning over standard prompting.

---

## Key Findings

- LLMs can effectively generate and select waypoints in real-time, leveraging multimodal scene understanding.
- The agent demonstrates strong local reasoning but struggles with long-horizon planning when the goal is outside its field of view.
- Most failures are due to the agent drifting away from the target, highlighting the need for richer memory or global guidance.
- Hardcoded controllers outperform LLMs for precise low-level control, but LLMs offer flexible high-level reasoning.

---

## Getting Started

### Prerequisites

- Unreal Engine 5 (for SimWorld)
- Python 3.8+
- OpenAI GPT-4o-mini (or compatible LLM API)
- See `requirements.txt` for dependencies

### Usage

1. **Set up SimWorld:** Launch the UE5 environment and initialize the simulation.
2. **Configure Agent:** Set agent starting and goal positions.
3. **Run Navigation Loop:** The agent will:
    - Capture current observations (RGB, depth, segmentation)
    - Prompt the LLM for candidate waypoints
    - Convert and select the optimal waypoint
    - Move toward the selected waypoint
    - Repeat until the goal is reached or a stopping condition is met

---

## Future Work

- **Dynamic environments:** Introduce moving obstacles (pedestrians, vehicles) and non-physical constraints (e.g., traffic lights).
- **Richer prompts:** Provide the LLM with orientation and heading information to improve reorientation and recovery.
- **Hierarchical planning:** Integrate global planning modules or memory to mitigate local-only reasoning.
- **Human-like navigation:** Teach societal rules (e.g., sidewalk preference), variable speed, and more natural movement patterns.

---
