# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains multiple components for Unitree Go2 quadruped robot reinforcement learning simulation and deployment:

- **`dreamwaq/`** – Main DreamWaQ algorithm implementation for rough‑terrain locomotion with Context‑aided Estimator Network (CENet).
- **`gym_basic/`** – Basic IsaacGym examples (camera, joint inspection).
- **`unitree_rl_gym/`** – Reference flat‑terrain locomotion implementation for Go2, H1, H1_2, G1 robots.
- **`wtw/`** – Sim‑to‑real deployment of “Walk These Ways” using Unitree SDK2 and LCM.

Each component is largely independent but shares robot URDFs and meshes from `dreamwaq/legged_gym/resources/`.

## Common Development Tasks

### Environment Setup

1. **Install IsaacGym Preview 4** (required for all components) from NVIDIA’s website.
2. **Install dependencies** per component:

   ```bash
   # For dreamwaq
   cd dreamwaq
   pip install -e ./rsl_rl
   pip install -e ./legged_gym
   pip install -r requirements.txt  # wandb, protobuf

   # For unitree_rl_gym
   cd unitree_rl_gym
   pip install -e .

   # For wtw
   cd wtw
   pip install -e .
   ```

   Note: `rsl_rl` and `legged_gym` are included as subdirectories; install them with `pip install -e .` in their respective folders.

3. **Set library path** if encountering `libpython` errors:
   ```bash
   export LD_LIBRARY_PATH=/path/to/conda/env/lib
   ```

### Training (dreamwaq)

Use the `train.py` script with a task name:

```bash
cd dreamwaq/legged_gym/scripts
python train.py --task=a1_waq --headless
```

**Task options** (see `dreamwaq/legged_gym/envs/a1/a1_config.py` and `go2/go2_config.py`):
- `--task=a1_base` – observation without linear velocity
- `--task=a1_oracle` – true linear velocity + privileged terrain info
- `--task=a1_waq` – estimated velocity + privileged info + observation history
- For Go2, analogous tasks: `go2_base`, `go2_oracle`, `go2_waq`, `go2_est`

**Common flags**:
- `--headless` – run without simulator GUI
- `--num_envs` – override number of parallel environments
- `--seed` – random seed
- `--max_iterations` – training iterations
- `--resume` – resume from checkpoint
- `--experiment_name`, `--run_name` – logging identifiers

### Inference / Playback

```bash
cd dreamwaq/legged_gym/scripts
python play.py --task=a1_waq --load_run=Sep04_14-24-54_waq --checkpoint=250
```

- `--load_run` – folder name inside `legged_gym/logs/<task_folder>/`
- `--checkpoint` – model checkpoint number (e.g., `250` for `model_250.pt`)

For a close‑up view of a single agent, use `mini_test.py` with the same arguments.

To export deployment-ready models with RMS normalization parameters embedded in ONNX metadata, set `EXPORT_POLICY = True` and related export flags in `dreamwaq/legged_gym/scripts/play.py`. RMS parameters will be embedded as JSON in the ONNX file's metadata_props under the key `dreamwaq.rms`.

### Testing

A simple environment test exists:

```bash
cd dreamwaq/legged_gym/tests
python test_env.py --task=a1_base
```

### Sim‑to‑Real Deployment (wtw)

Deployment requires LCM and Unitree SDK2:

1. **Compile LCM** and **install SDK2** (see `wtw/README.md`).
2. **Build the bridge**:
   ```bash
   cd wtw/go2_gym_deploy
   rm -rf build && mkdir build && cd build
   cmake .. && make -j
   ```
3. **Run the bridge** (replace `eth0` with your network interface):
   ```bash
   sudo ./lcm_position_go2 eth0
   ```
4. **Launch the policy**:
   ```bash
   cd wtw/go2_gym_deploy/scripts
   python deploy_policy.py
   ```
   Use R2 to start the controller; L2+B for damping (emergency stop).

### Basic Examples (gym_basic)

```bash
cd gym_basic
python go2_cam.py          # egocentric camera demo
python go2_inspection.py   # joint state inspection
```

## High‑Level Architecture

### dreamwaq

- **Environment** (`legged_gym/`): Based on `leggedrobotics/legged_gym`. Contains robot configurations (`a1_config.py`, `go2_config.py`), base environment class (`legged_robot.py`), terrain generation (`terrain.py`), and task registry (`task_registry.py`).
- **RL Algorithm** (`rsl_rl/`): Based on `leggedrobotics/rsl_rl`. Implements PPO (`ppo.py`), actor‑critic networks (`actor_critic.py`), and training loops (`on_policy_runner.py`).
- **CENet** (`rsl_rl/vae/cenet.py`): Context‑aided Estimator Network that estimates linear velocity from observation history and privileged terrain information. Used in DreamWaQ tasks (`a1_waq`, `go2_waq`).
- **Runner variants**: `OnPolicyRunner` (base), `OnPolicyRunnerWaq` (DreamWaQ), `OnPolicyRunnerEst` (estimator‑only). Selected automatically based on task configuration.

**Key configuration files**:
- Robot‑specific configs: `dreamwaq/legged_gym/envs/{a1,go2}/*_config.py`
- Base config classes: `dreamwaq/legged_gym/envs/base/legged_robot_config.py`
- Training configs: `dreamwaq/rsl_rl/rsl_rl/runners/on_policy_runner.py`

### unitree_rl_gym

Flat‑terrain reference implementation. Shares the same `legged_gym` structure but with configurations for Go2, H1, H1_2, G1. Use `--task=go2` (or `h1`, `h1_2`, `g1`) with the provided `train.py`.

### wtw (Walk These Ways)

- **Training environment** (`go2_gym/`): Custom environment for “Walk These Ways” algorithm.
- **Deployment stack** (`go2_gym_deploy/`): LCM communication bridge (`lcm_position_go2.cpp`), SDK2 integration, and deployment script (`deploy_policy.py`).
- **Important**: Robot configuration changes should be made in `wtw/go2_gym/envs/go2/go2_config.py`, not in the base legged_robot_config.

## Important Notes

- **IsaacGym version**: Preview 4 is required; the code is not compatible with newer IsaacGym releases.
- **CUDA / driver**: CUDA 12.1+ driver recommended. For Jetson deployment, follow the JetPack / CUDA instructions in `wtw/README.md`.
- **Logging**: Training logs are saved under `dreamwaq/legged_gym/logs/`. WandB integration is optional (set `WANDB = True` in `train.py`).
- **Cross‑computer inference**: Copy `model_<N>.pt` to the same relative path on the target machine and use `play.py` with the same `--task`, `--load_run`, and `--checkpoint`.
- **Configuration precedence**: Command‑line arguments override config‑file values; config‑file values override default class values.

## File Naming and Location Patterns

- Robot URDFs and meshes: `dreamwaq/legged_gym/resources/robots/{a1,go2,...}/`
- Training scripts: `*/legged_gym/scripts/train.py`
- Inference scripts: `*/legged_gym/scripts/play.py` and `mini_test.py`
- Environment registration: `*/legged_gym/envs/__init__.py`
- Task‑specific configs: `*/legged_gym/envs/<robot>/<robot>_config.py`
- CENet implementation: `dreamwaq/rsl_rl/rsl_rl/vae/cenet.py`
- LCM bridge: `wtw/go2_gym_deploy/unitree_sdk2_bin/lcm_position_go2.cpp`

When modifying robot parameters, always check the robot‑specific config file first; base configs are inherited but often overridden.
