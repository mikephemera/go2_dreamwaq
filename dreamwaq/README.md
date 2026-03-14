# dreamwaq

> Independent implementation of [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** [Jungyeon Lee (curieuxjy)](https://github.com/curieuxjy)

I independently implemented the DreamWaQ algorithm based on the paper. The core component, **Context-aided Estimator Network (CENet)**, has been carefully implemented and verified to work as described. Feel free to explore the code and experiment with it!

---

https://github.com/curieuxjy/dreamwaq/assets/40867411/5dcea5c9-3ff3-469d-baa7-70f0852a0395

[🎥 1080 Streaming Video in YouTube](https://youtu.be/5rwFcz-lerw)

---

## Table of Contents

| Section                                     | Description                                             |
| ------------------------------------------- | ------------------------------------------------------- |
| [Start Manual](#start-manual)               | Project environment setup and execution instructions    |
| [Main Code Structure](#main-code-structure) | Main code structure explanation                         |
| [Result Graphs](#result-graphs)             | Training result graphs                                  |
| [Result Motions](#result-motions)           | Training result walking motion videos (gif per section) |

---

## Start Manual

### Start **w/o** this repository

> This is the initial setup for implementation project independent of this repository. To run based on this repository, please refer to the w/ execution steps below.

1. Install IsaacGym ver.4
2. Download [rsl-rl](https://github.com/leggedrobotics/rsl_rl) from github as **zip** file and install `pip install -e .`
3. Download [legged-gym](https://github.com/leggedrobotics/legged_gym) from github as **zip** file and install `pip install -e .`
4. Modify some experiment logging parts including wandb (must login with your own account)

---

### Start **w/** this repository

> Please follow the steps below when starting the project based on this repository.

1. Install IsaacGym ver.4 [isaac-gym page](https://developer.nvidia.com/isaac-gym)
2. Run `pip install -e .` in `rsl-rl/` directory
3. Run `pip install -e .` in `legged-gym/` directory
4. `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`
   - `export LD_LIBRARY_PATH=/home/jungyeon/anaconda3/envs/go2/lib`
5. `pip install tensorboard wandb opencv-python`
6. `AttributeError: module 'distutils' has no attribute 'version'`
   - `pip install setuptools==59.5.0`
   - (ref) https://github.com/pytorch/pytorch/issues/69894
7. Start rough-terrain locomotion training with A1 or Go2 (refer to table below)

#### Task Options

| Option              | Config              | Critic Obs | Actor Obs | Memo                                                             |
| ------------------- | ------------------- | :--------: | :-------: | ---------------------------------------------------------------- |
| `--task=a1_base`    | `A1RoughBaseCfg`    |     45     |    45     | blind base policy (no linear velocity input)                     |
| `--task=a1_oracle`  | `A1RoughOracleCfg`  |    238     |    238    | true linear velocity + privileged terrain info                   |
| `--task=a1_waq`     | `A1RoughWaqCfg`     |    238     |    64     | estimated linear velocity + context vector + observation history |
| `--task=a1_est`     | `A1RoughEstCfg`     |    238     |    48     | estimated linear velocity (Estimator-only variant)               |
| `--task=go2_base`   | `Go2RoughBaseCfg`   |     45     |    45     | Go2 blind base policy                                            |
| `--task=go2_oracle` | `Go2RoughOracleCfg` |    238     |    238    | Go2 oracle policy with privileged terrain info                   |
| `--task=go2_waq`    | `Go2RoughWaqCfg`    |    238     |    64     | Go2 DreamWaQ policy (estimated velocity + context)               |

> Note: `go2_est` config classes exist, but `--task=go2_est` is not currently registered in `legged_gym/envs/__init__.py`.

---

### Start **w/** docker

> Please follow the steps below when starting via docker based on this repository.
> A driver supporting CUDA 12.1 or higher must be installed.

1. Download [Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4](https://developer.nvidia.com/isaac-gym)
   You may also need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   > NVIDIA Container Toolkit DOES NOT work with Docker Desktop., only work with [Docker Engine](https://docs.docker.com/engine/install/)... Then you may need add user to docker to solve permission issue without root. see [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/)
2. Move the downloaded `IsaacGym_Preview_4_Package.tar.gz` file to `asset/IsaacGym_Preview_4_Package.tar.gz`
3. Build docker with the following command:
   ```bash
   cd /mnt/datafiles/Work-syncfree/go2_dreamwaq/dreamwaq
   docker build . -t dreamwaq/dreamwaq -f docker/Dockerfile  --build-arg UID=$(id -u) --build-arg GID=$(id -g)
   ```
   > The original docker file use [script](https://raw.githubusercontent.com/JeiKeiLim/my_term/main/run.sh) which has a bug on neovim install. I fixed it using dreamwaq/docker/run_fixed.sh.
4. Run docker with the following command:

   ```bash
   cd /mnt/datafiles/Work-syncfree/go2_dreamwaq/
   # check display before run
   echo $DISPLAY
   # create logs directory and run container with name and volume

    docker run -ti --privileged \
       -e DISPLAY=:1 \
       -e TERM=xterm-256color \
       -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
       --network host \
       -v $PWD/dreamwaq:/home/user/dreamwaq \
       -v $PWD/dreamwaq_logs:/home/user/legged_gym/logs \
       --name dreamwaq-run \
       --gpus all \
       dreamwaq/dreamwaq /usr/bin/zsh
   ```

   <del>
   > Docker env has another issue, PyTorch requires GCC 9+, but the system default gcc command points to GCC 8.4.0. Here is the solution:</del>
   <del>
   #1. Update alternatives configuration
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80
   </del>
   <del>
   #2. Choose GCC 9 as default
   sudo update-alternatives --config gcc
   sudo update-alternatives --config g++
   </del>
   <del>
   #3. verify
   gcc --version
   #It should show gcc version 9.x.x
   </del>
   <del>
   #4. clear pytorch cache
   rm -rf /home/user/.cache/torch_extensions/py38_cu121
   rm -rf /home/user/.cache/torch_extensions/*
   </del>

---

### Command

Execute within the container:

```bash
cd dreamwaq/legged_gym/legged_gym/scripts
python train.py --task=a1_waq --headless
# Use --headless to avoid memcpy error when display not connected.
```

#### Training

```bash
python train.py --task=[TASK_NAME] --headless
```

- `--headless`: Option to run training without opening simulator window. Add this option when running on a server without display.

#### Inferencing

```bash
python play.py --task=[TASK_NAME] --load_run=[LOAD_FOLDER] --checkpoint=[CHECKPOINT_NUMBER]
```

| Parameter             | Description                                        | Example                              |
| --------------------- | -------------------------------------------------- | ------------------------------------ |
| `[LOAD_FOLDER]`       | Folder name inside `legged_gym/logs/[task folder]` | `Sep04_14-24-54_waq`                 |
| `[task folder]`       | Task-specific log folder                           | `rough_a1/rough_a1_waq/rough_a1_est` |
| `[CHECKPOINT_NUMBER]` | Number of **model\_[NUMBER].pt** file              | `250`                                |

**Complete command example:**

```bash
python play.py --task=a1_waq --load_run=Sep04_14-24-54_waq --checkpoint=250
```

- Inferencing code to view a single agent up close: `mini_test.py` (options same as `play.py`)
- There are adjustable options in the main loop of each inferencing script, adjust True/False as needed.

#### Cross-computer Inference

If you want to inference a **model\_[NUMBER].pt** file trained on a different computer:

| Step | Computer A (Training)       | Computer B (Inferencing)                                                               |
| ---- | --------------------------- | -------------------------------------------------------------------------------------- |
| 1    | -                           | Create a new folder named `FOLDER_NAME` in `legged_gym/logs/[task folder]`             |
| 2    | Copy **model\_[NUMBER].pt** | Paste to `FOLDER_NAME`                                                                 |
| 3    | -                           | Run `python play.py --task=[TASK_NAME] --load_run=[FOLDER_NAME] --checkpoint=[NUMBER]` |

```bash
# Use tensorboard
tensorboard --logdir=/mnt/datafiles/Work-syncfree/go2_dreamwaq/dreamwaq_logs

# Workflow reuse docker container
docker start dreamwaq-run
docker exec -it dreamwaq-run /bin/bash
cd dreamwaq/legged_gym/legged_gym/scripts
# python train.py --task=go2_waq --headless
python play.py --task=go2_waq --load_run=Mar09_15-10-01_waq --checkpoint=5000 #Need to embed RMS
python play.py --task=go2_waq --load_run=Mar13_20-46-45_waq --checkpoint=5000 #No RMS
python mini_test.py --task=go2_waq --num_envs 1

#cd out
python dreamwaq/legged_gym/legged_gym/scripts/read_onnx_rms.py legged_gym/logs/rough_go2_waq/exported/policies/policy.onnx
```

---
## 观测，缩放和关节顺序
`base/waq/est` 会去掉 `base_lin_vel`，而 `oracle` 保留。

1. **观测顺序（env 输出）**

- 代码位置：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:346`
- 拼接来源是 `obs_dict` 按插入顺序 `torch.cat(list(obs_dict.values()))`：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:427`
- `go2_base/go2_waq/go2_est` 会删掉 `base_lin_vel`：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:423`
- 下述索引区间使用 Python slice 语义，即 `start:end` 表示包含 `start`、不包含 `end`。

`go2_base` / `go2_waq`（45维）顺序：

- `0:3` `base_ang_vel` (3) # 基座角速度
- `3:6` `projected_gravity` (3) # 投影重力
- `6:9` `commands[:3]` (3) # 指令
- `9:21` `dof_pos - default_dof_pos` (12) # 关节位置
- `21:33` `dof_vel` (12) # 关节速度
- `33:45` `actions` (12) # 上一时刻动作

`go2_oracle`（48维）顺序：

- `0:3` `base_lin_vel` (3)
- `3:6` `base_ang_vel` (3)
- `6:9` `projected_gravity` (3)
- `9:12` `commands[:3]` (3)
- `12:24` `dof_pos - default_dof_pos` (12)
- `24:36` `dof_vel` (12)
- `36:48` `actions` (12)

2. **训练时真正喂给策略的输入（尤其 `go2_waq`）**

- `go2_waq` 里 actor 输入不是纯 45 维，而是：`[obs(45), vel_input(3), context_vec(16)]`
- 代码：`dreamwaq/rsl_rl/rsl_rl/runners/on_policy_runner.py:559`
- 所以 `go2_waq` actor 输入维度是 64（配置里也对应 `num_context=16`, `num_estvel=3`）：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:233-235`

3. **缩放（scale）**

- 固定缩放参数定义在：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot_config.py:201`
  - `lin_vel=2.0` # 线速度
  - `ang_vel=0.25` # 角速度
  - `dof_pos=1.0` # 关节位置
  - `dof_vel=0.05` # 关节速度
  - `height_measurements=5.0` # 高度测量
- 但是否应用固定缩放取决于 `fixed_norm`。`go2_waq` 显式 `fixed_norm=False`：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:244`
- `fixed_norm=False` 时，上述固定 scale 不乘到 obs 上（只在 `fixed_norm=True` 分支才乘）：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:415`
- 训练中主要使用 **RMS 标准化**（`obs_rms=True` 等）：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:263`，实际标准化代码在：`dreamwaq/rsl_rl/rsl_rl/runners/on_policy_runner.py:502`

4. **关节顺序（DOF 顺序）**

- DOF 顺序来自 IsaacGym 读取 asset 的 `dof_names`，不是 `default_joint_angles` 字典顺序：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:1199`
- 当前 go2.urdf 中 12 个 revolute joint 顺序是：
      1. FL_hip_joint   （左前髋）
      2. FL_thigh_joint （左前大腿）
      3. FL_calf_joint  （左前小腿）
      4. FR_hip_joint   （右前髋）
      5. FR_thigh_joint （右前大腿）
      6. FR_calf_joint  （右前小腿）
      7. RL_hip_joint   （左后髋）
      8. RL_thigh_joint （左后大腿）
      9. RL_calf_joint  （左后小腿）
      10. RR_hip_joint   （右后髋）
      11. RR_thigh_joint （右后大腿）
      12. RR_calf_joint  （右后小腿）
- 依据：
`dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf:157`、`dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf:416`、`dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf:675`、`dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf:934`

## 导出部署时（ONNX/TorchScript）输入维度与字段顺序对照

### 1) 导出文件接口（直接对应 `export.py`）

| Model file    | Format      | Input name    | Input shape            | Output name              | Output shape        | Notes                                                                                               |
| ------------- | ----------- | ------------- | ---------------------- | ------------------------ | ------------------- | --------------------------------------------------------------------------------------------------- |
| `policy_1.pt` | TorchScript | N/A           | `[B, actor_input_dim]` | return tensor            | `[B, 12]`           | TorchScript module from actor network; unlike ONNX, input name is not explicitly set in `export.py` |
| `policy.onnx` | ONNX        | `input`       | `[B, actor_input_dim]` | `output`                 | `[B, 12]`           | Exported with dynamic batch axis                                                                    |
| `cenet.onnx`  | ONNX        | `obs_history` | `[B, 225]`             | `est_vel`, `context_vec` | `[B, 3]`, `[B, 16]` | For WAQ deployment                                                                                  |
| `estnet.onnx` | ONNX        | `obs_history` | `[B, 225]`             | `est_vel`                | `[B, 3]`            | For EST deployment                                                                                  |

**RMS 归一化参数**：训练时使用的 RMS 归一化参数（obs_rms, true_vel_rms, privileged_obs_rms）已嵌入到 ONNX 文件的元数据中（metadata_props），键为 `dreamwaq.rms`。导出时默认启用该嵌入（`embed_rms_in_onnx=True`）

`obs_history=225` comes from `len_obs_history(5) * num_observations(45)`.

### 2) Go2 task -> policy input dimension

| Task                      | Actor input tensor to policy       | `actor_input_dim`  |
| ------------------------- | ---------------------------------- | ------------------ |
| `go2_base`                | `obs`                              | `45`               |
| `go2_waq`                 | `cat(obs, vel_input, context_vec)` | `45 + 3 + 16 = 64` |
| `go2_oracle`              | `cat(obs, privileged_obs)`         | `48 + 190 = 238`   |
| `go2_est` (if registered) | `cat(obs, est_vel)`                | `45 + 3 = 48`      |

### 3) `obs` 字段顺序（45/48 维）

`go2_base` / `go2_waq` / `go2_est` uses 45-dim `obs`:

| Index range | Field                       | Dim |
| ----------- | --------------------------- | --- |
| `0:3`       | `base_ang_vel`              | 3   |
| `3:6`       | `projected_gravity`         | 3   |
| `6:9`       | `commands[:3]`              | 3   |
| `9:21`      | `dof_pos - default_dof_pos` | 12  |
| `21:33`     | `dof_vel`                   | 12  |
| `33:45`     | `actions`                   | 12  |

`go2_oracle` uses 48-dim `obs`:

| Index range | Field                       | Dim |
| ----------- | --------------------------- | --- |
| `0:3`       | `base_lin_vel`              | 3   |
| `3:6`       | `base_ang_vel`              | 3   |
| `6:9`       | `projected_gravity`         | 3   |
| `9:12`      | `commands[:3]`              | 3   |
| `12:24`     | `dof_pos - default_dof_pos` | 12  |
| `24:36`     | `dof_vel`                   | 12  |
| `36:48`     | `actions`                   | 12  |

### 4) `privileged_obs` 字段顺序（oracle/critic 侧 190 维）

| Index range | Field           | Dim |
| ----------- | --------------- | --- |
| `0:3`       | `disturb_force` | 3   |
| `3:190`     | `heights`       | 187 |

For `go2_oracle` policy input (`238`), the concat order is:

`[obs(48), privileged_obs(190)]` -> `disturb_force` starts at global index `48`, `heights` starts at `51`.

### 5) `obs_history` 展平顺序（给 CENet/ESTNet）

`get_observation_history()` keeps a FIFO queue by:

`obs_history_buf = cat(obs_history_buf[:, 1:], obs_buf.unsqueeze(1), dim=1)`

Then `reshape(num_envs, -1)` flattens time-first in memory order, so layout is:

`[o(t-4), o(t-3), o(t-2), o(t-1), o(t)]`, each `o(*)` is the same 45-dim order listed above.

For WAQ deployment, recommended runtime graph is:

`obs_history(225) -> cenet.onnx -> (est_vel(3), context_vec(16))`

`actor_input = cat(obs(45), est_vel(3), context_vec(16)) -> policy.onnx`

## Main Code Structure

- Explanation of important files in the project code. Files related to the robot platform and algorithms used in the project were selected. Please refer to the description next to each file name.
  - Robot platform used (environment): A1
  - Learning algorithm used: PPO

```
dreamwaq
│
├── legged_gym
│   ├── legged_gym
│   │   ├── envs
│   │   │   ├── __init__.py: Environment registration for training execution. Referenced by task_registry.
│   │   │   ├── a1/a1_config.py: Variable classes for A1 platform. Inherits from legged_robot_config.py classes.
│   │   │   └── base
│   │   │        ├── legged_robot.py: Base environment class for locomotion task. LeggedRobot Class
│   │   │        └── legged_robot_config.py: Variable classes for LeggedRobot. LeggedRobotCfg Class / LeggedRobotCfgPPO Class
│   │   ├── scripts
│   │   │   ├── train.py: Main training execution code. wandb settings setup. (Refer to Command-training)
│   │   │   ├── play.py: Code to check walking inference motion of multiple agents on various terrains after training. (Refer to Command-inference)
│   │   │   └── mini_test.py: Code to check walking inference motion of multiple agents on various terrains after training. (Refer to Command-inference)
│   │   └── utils
│   │       ├── logger.py: Code for matplotlib plot used in play.py and mini_test.py.
│   │       ├── task_registry.py: Connects environment and algorithm based on training environment info registered in envs/__init__.py.
│   │       └── terrain.py: Terrain class for walking. Referenced by LeggedRobot.
│   │
│   └── resources/robots/a1: Robot platform information (urdf&mesh)
│
└── rsl_rl
    └── rsl_rl
        ├── algorithms
        │   └── ppo.py: PPO algorithm code. Uses Actor/Critic classes from actor_critic.py.
        ├── modules
        │   └── actor_critic.py: Actor/Critic class code.
        ├── runners
        │   └── on_policy_runner.py: File containing OnPolicyRunner class with the main RL loop (learn function).
        │                            Base model uses OnPolicyRunner class, DreamWaQ model uses OnPolicyRunnerWaq class,
        │                            Estnet model uses OnPolicyRunnerEst class for training code execution.
        │                            (Classes are distinguished by modifications at the stage before the RL main loop [before actor/critic network stage])
        ├── utils
        │   └── rms.py: Running Mean Std class for CENet's normal prior distribution training.
        └── vae
            ├── cenet.py: Context-Aided Estimator Network (CENet) class.
            └── estnet.py: Estimator class for comparison model group.

```

---

## Result Graphs

Reward Graph for approximately 1000 iterations of training

![](./asset/two_models_rew.png)

### DreamWaQ model

- State plot of 1 robot agent after training
  - Row 1: Plot of x, y direction velocity and yaw direction command vs actual measured physical quantities from base state
  - Row 2: Plot of estimated velocity through CENet vs true velocity measured from simulator
  - Row 3: Error plot between estimated velocity and true velocity
    - Column 1: Squared error of each x, y, z direction component
    - Column 2, 3: Mean squared error of x, y directions

![](./asset/a1_waq_est_vel.png)

### Base model

- State plot of 1 robot agent after training (Unlike DreamWaQ, there is no estimated velocity, so the plotted graphs are different.)
  - Row 1: Plot of x, y direction velocity and yaw direction command vs actual measured physical quantities from base state
  - Row 2 Column 1/2: Position and velocity of 1 joint
  - Row 2 Column 3: Base z direction velocity
  - Row 3 Column 1: Contact force of 4 feet
  - Row 3 Column 2/3: Torque of 1 joint

![](./asset/a1_base_no_vel.png)

---

## Result Motions

> **Notice:** The videos below were recorded using the **A1 platform**. However, this repository also includes code for applying the algorithm to the **Go2 platform**.

### Walking Performance of a Reproduction Model in Different Terrains

- Smooth Slope / Rough Slope

![](./asset/1.gif)

- Stair Up / Stair Down

![](./asset/2.gif)

- Discrete / Mixed

![](./asset/3.gif)

---

### Comparative Analysis of Walking Motion Between the Reproduction Model and the Base Model

> small difference: naturalness of motion
>
> big difference: foot stuck / unstable step

- Smooth Slope(small difference)

![](./asset/4.gif)

- Rough Slope(small difference)

![](./asset/5.gif)

- Stair Up(big difference)

![](./asset/6.gif)

- Stair Down(big difference)

![](./asset/7.gif)

- Discrete(big difference)

![](./asset/8.gif)
