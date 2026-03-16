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
# Python 实际导入路径可能不是当前工作区，而是容器里旧副本。 go2_waq 的 RMS 开关取决于“被导入的那份配置文件”，不是你正在编辑器里打开的那份。有可能出现“源码里是 false，但 train_cfg.yaml 还是 true”
# 运行check_dreamwaq_env.sh脚本可以检查 legged_gym 导入路径是否来自 /home/user/dreamwaq/legged_gym
# 检查 go2_waq 的 obs_rms / privileged_obs_rms / true_vel_rms 是否全为 False
# 带 --fix 时，自动重装 editable 包到挂载目录，再次校验，避免手工漏步骤。然后再训练
# python train.py --task=go2_waq --headless
#python play.py --task=go2_waq --load_run=Mar09_15-10-01_waq --checkpoint=5000 #Need to embed RMS
python play.py --task=go2_waq --load_run=Mar14_19-20-17_waq --checkpoint=10000 #Need to embed RMS
python play.py --task=go2_waq --load_run=Mar14_16-57-53_waq --checkpoint=5000 #No RMS
python mini_test.py --task=go2_waq --num_envs 1

#cd out
python dreamwaq/legged_gym/legged_gym/scripts/read_onnx_rms.py legged_gym/logs/rough_go2_waq/exported/policies/policy.onnx
```

## 1. 观测、缩放与关节顺序

### 1.1 观测顺序（env 输出）

- **代码位置**：
  - `dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:346`
  - 拼接来源是 `obs_dict` 按插入顺序 `torch.cat(list(obs_dict.values()))`：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:427`
- `go2_base` / `go2_waq` / `go2_est` 会删掉 `base_lin_vel`：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:423`
- 下述索引区间使用 Python slice 语义，即 `start:end` 表示包含 `start`、不包含 `end`。

#### `go2_base` / `go2_waq`（45 维）顺序

| 索引范围 | 字段                     | 维度 | 物理意义 |
|----------|--------------------------|------|----------|
| `0:3`    | `base_ang_vel`           | 3    | 基座角速度（滚转、俯仰、偏航） |
| `3:6`    | `projected_gravity`      | 3    | 投影重力向量（机体坐标系下的重力方向） |
| `6:9`    | `commands[:3]`           | 3    | 用户指令（通常为 `[vx, vy, ωz]`） |
| `9:21`   | `dof_pos - default_dof_pos` | 12 | 关节位置相对于默认位置的偏差 |
| `21:33`  | `dof_vel`                | 12   | 关节速度 |
| `33:45`  | `actions`                | 12   | 上一时刻输出的关节位置指令 |

#### `go2_oracle`（48 维）顺序

| 索引范围 | 字段                     | 维度 |
|----------|--------------------------|------|
| `0:3`    | `base_lin_vel`           | 3    |
| `3:6`    | `base_ang_vel`           | 3    |
| `6:9`    | `projected_gravity`      | 3    |
| `9:12`   | `commands[:3]`           | 3    |
| `12:24`  | `dof_pos - default_dof_pos` | 12 |
| `24:36`  | `dof_vel`                | 12   |
| `36:48`  | `actions`                | 12   |

### 1.2 训练时真正喂给策略的输入

- `go2_waq` 任务的 Actor 输入不是纯 45 维观测，而是 **`[obs(45), vel_input(3), context_vec(16)]`**。
- **代码位置**：`dreamwaq/rsl_rl/rsl_rl/runners/on_policy_runner.py:559`
- 因此 `go2_waq` 的 Actor 输入维度为 **64**（配置中对应 `num_context=16`, `num_estvel=3`）：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:233-235`

### 1.3 CENet 与 `context_vec` 维度含义

#### 定义与代码位置
- **`num_context=16`** 定义在训练配置文件 `dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py`（第234行）的 `Go2RoughWaqCfg` 类中：
  ```python
  class env(Go2RoughBaseCfg.env):
      num_observations = 45  # o(45)
      len_obs_history = 5  # o_H
      num_context = 16  # z
      num_estvel = 3  # v
      num_privileged_obs = 190  # d(3) + h(187)
  ```

#### 网络架构与生成机制
- **CENet 结构**：CENet 是一个变分自编码器（VAE），其编码器输出维度为 `latent_dim1=35`（对应 `3 + 16×2`，其中3为 `est_vel`，16×2 分别为 `mu` 和 `logvar`）。
- **生成过程**：
  1. 编码器从 5 帧历史观测（225维）提取特征。
  2. 输出分为两部分：`est_vel`（3维）和 `context_vec_params`（32维）。
  3. `context_vec_params` 被分割为 16 维均值（`mu`）和 16 维对数方差（`logvar`）。
  4. 通过重参数化技巧 `context_vec = mu + eps * exp(logvar/2)` 得到 16 维的 `context_vec`（潜在向量 `z`）。
  5. 最终潜在表示为 `latent = [est_vel(3), context_vec(16)]`（19维）。

#### 为什么是 16 维？
- **设计依据**：基于 DreamWaQ 论文（arXiv:2301.10602）提出的 Context‑Aided Estimator Network（CENet）架构。
- **超参数调优**：16 维潜在空间是经验性选择，平衡了以下因素：
  - **表示能力**：足够编码地形、外部扰动等环境上下文信息。
  - **训练稳定性**：避免过高的维度导致训练困难或过拟合。
  - **计算效率**：与 3 维 `est_vel` 拼接后形成 19 维潜在表示，作为解码器输入。
- **网络维度匹配**：在 CENet 实现（`dreamwaq/rsl_rl/rsl_rl/vae/cenet.py`）中，编码器输出 `latent_dim1=35`（`3 + 16×2`），解码器输入 `latent_dim2=19`（`3 + 16`），确保了维度一致性。

#### 作用与训练‑部署对齐
- **作用**：`context_vec` 编码了环境上下文信息（地形特征、外部扰动等），与当前观测 `obs` 和估计速度 `est_vel` 拼接后，为策略网络提供更丰富的环境表征，提升在崎岖地形上的适应能力。
- **训练**：CENet 通过 VAE 的变分下界（ELBO）学习从历史观测中推断 `context_vec`，同时鼓励潜在空间具有良好的结构（如平滑性、解耦性）。
- **部署**：CENet 根据实时观测历史输出相同的 16 维向量，确保策略在未见过的环境中也能利用学习到的上下文表示进行适应。

### 1.4 缩放（scale）

- **固定缩放参数**定义在 `dreamwaq/legged_gym/legged_gym/envs/base/legged_robot_config.py:201`：
  - `lin_vel=2.0`      # 线速度
  - `ang_vel=0.25`     # 角速度
  - `dof_pos=1.0`      # 关节位置
  - `dof_vel=0.05`     # 关节速度
  - `height_measurements=5.0` # 高度测量

- **是否应用固定缩放**取决于 `fixed_norm`。`go2_waq` 显式设置为 `fixed_norm=False`：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:244`
  - 当 `fixed_norm=False` 时，上述固定缩放 **不** 乘到观测上（仅在 `fixed_norm=True` 分支才乘）：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:415`

- **训练中主要使用 RMS 标准化**（`obs_rms=True` 等）：`dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py:263`
  - 实际标准化代码在：`dreamwaq/rsl_rl/rsl_rl/runners/on_policy_runner.py:502`
  - **RMS 参数提取**需要 rsl 包，当前环境没有，需前往训练代码目录环境提取，然后硬编码回来。

### 1.5 关节顺序（DOF 顺序）

- DOF 顺序来自 IsaacGym 读取 asset 的 `dof_names`，**不是** `default_joint_angles` 字典顺序：`dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py:1199`
- 当前 `go2.urdf` 中 12 个 revolute joint 顺序如下（依据 `dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf` 对应行号）：

| 序号 | 关节名（英文） | 中文说明 |
|------|----------------|----------|
| 1    | `FL_hip_joint`   | 左前髋 |
| 2    | `FL_thigh_joint` | 左前大腿 |
| 3    | `FL_calf_joint`  | 左前小腿 |
| 4    | `FR_hip_joint`   | 右前髋 |
| 5    | `FR_thigh_joint` | 右前大腿 |
| 6    | `FR_calf_joint`  | 右前小腿 |
| 7    | `RL_hip_joint`   | 左后髋 |
| 8    | `RL_thigh_joint` | 左后大腿 |
| 9    | `RL_calf_joint`  | 左后小腿 |
| 10   | `RR_hip_joint`   | 右后髋 |
| 11   | `RR_thigh_joint` | 右后大腿 |
| 12   | `RR_calf_joint`  | 右后小腿 |

---

## 2. 导出部署接口（ONNX/TorchScript）

### 2.1 导出文件接口（直接对应 `export.py`）

| 模型文件 | 输入名 | 输入形状 | 输出名 | 输出形状 | 备注 |
|----------|--------|----------|--------|----------|------|
| `policy_1.pt` | N/A | `[B, actor_input_dim]` | return tensor | `[B, 12]` | TorchScript 模块（输入名未显式设置） |
| `policy.onnx` | `input` | `[1, actor_input_dim]` | `output` | `[1, 12]` | ~~动态 batch 轴~~改为固定1维（静态batch） |
| `cenet.onnx`  | `obs_history` | `[1, 225]` | `est_vel`, `context_vec` | `[1, 3]`, `[1, 16]` | 用于 WAQ 部署（静态batch） |
| `estnet.onnx` | `obs_history` | `[1, 225]` | `est_vel` | `[1, 3]` | 用于 EST 部署（静态batch） |

> **注意**：导出的 ONNX 模型（`policy.onnx`、`cenet.onnx`、`estnet.onnx`）**均已改为静态 batch 维度**（batch size=1）。在 `export.py` 脚本中，`dynamic_axes` 参数被注释掉，因此输入形状固定为 `[1, ...]`，输出形状同理。部署时无需支持动态 batch。

- `obs_history=225` 来源于 `len_obs_history(5) * num_observations(45)`。

### 2.2 Go2 任务 → 策略输入维度

| 任务 | Actor 输入张量 | `actor_input_dim` |
|------|----------------|-------------------|
| `go2_base` | `obs` | 45 |
| `go2_waq` | `cat(obs, vel_input, context_vec)` | 45 + 3 + 16 = **64** |
| `go2_oracle` | `cat(obs, privileged_obs)` | 48 + 190 = **238** |
| `go2_est`（如已注册） | `cat(obs, est_vel)` | 45 + 3 = **48** |

### 2.3 `obs` 字段顺序（45/48 维）

#### `go2_base` / `go2_waq` / `go2_est` 使用的 45 维 `obs`

| 索引范围 | 字段 | 维度 |
|----------|------|------|
| `0:3` | `base_ang_vel` | 3 |
| `3:6` | `projected_gravity` | 3 |
| `6:9` | `commands[:3]` | 3 |
| `9:21` | `dof_pos - default_dof_pos` | 12 |
| `21:33` | `dof_vel` | 12 |
| `33:45` | `actions` | 12 |

#### `go2_oracle` 使用的 48 维 `obs`

| 索引范围 | 字段 | 维度 |
|----------|------|------|
| `0:3` | `base_lin_vel` | 3 |
| `3:6` | `base_ang_vel` | 3 |
| `6:9` | `projected_gravity` | 3 |
| `9:12` | `commands[:3]` | 3 |
| `12:24` | `dof_pos - default_dof_pos` | 12 |
| `24:36` | `dof_vel` | 12 |
| `36:48` | `actions` | 12 |

### 2.4 `privileged_obs` 字段顺序（190 维）

| 索引范围 | 字段 | 维度 |
|----------|------|------|
| `0:3` | `disturb_force` | 3 |
| `3:190` | `heights` | 187 |

对于 `go2_oracle` 策略输入（238 维），拼接顺序为：
`[obs(48), privileged_obs(190)]` → `disturb_force` 从全局索引 48 开始，`heights` 从全局索引 51 开始。

### 2.5 `obs_history` 展平顺序（给 CENet/ESTNet）

- `get_observation_history()` 通过以下方式维护 FIFO 队列：
  ```python
  obs_history_buf = cat(obs_history_buf[:, 1:], obs_buf.unsqueeze(1), dim=1)
  ```
- 然后 `reshape(num_envs, -1)` 按 **时间优先**（time‑first）的内存顺序展平，布局为：
  ```
  [o(t-4), o(t-3), o(t-2), o(t-1), o(t)]
  ```
  每个 `o(*)` 均为上述 45 维观测顺序。

- **WAQ 部署推荐运行图**：
  ```
  obs_history(225) → cenet.onnx → (est_vel(3), context_vec(16))
  actor_input = cat(obs(45), est_vel(3), context_vec(16)) → policy.onnx
  ```

---

## 3. 推理流程详解

在 `CtrlEnv.step()`（`unitree_sim2x/ctrl_envs/crtl_env.py:237‑298`）中，每一步控制循环按以下顺序执行：

### 3.1 CENet 推理步骤

1. **观测归一化**（若配置了 `cenet_obs_rms_mean`）：
   ```python
   obs = (obs - self.cenet_obs_rms_mean[None, :]) / np.sqrt(self.cenet_obs_rms_var[None, :] + 1e-8)
   ```

2. **更新历史缓冲区**（5 帧滑动窗口）：
   ```python
   self.obs_history_buffer[:-1] = self.obs_history_buffer[1:]
   self.obs_history_buffer[-1] = obs[0]  # obs 形状为 (1, obs_dim)
   ```

3. **展平历史观测** → 形状 `(1, 225)`：
   ```python
   obs_history_flat = self.obs_history_buffer.flatten()[np.newaxis, :]
   ```

4. **调用 CENet**，获取 `est_vel`（3 维）和 `context_vec`（16 维）：
   ```python
   cenet_outputs = self.cenet_policy(obs_history_flat)
   est_vel = cenet_outputs["est_vel"]      # shape (1, 3)
   context_vec = cenet_outputs["context_vec"]  # shape (1, 16)
   ```

5. **拼接当前观测 + CENet 输出** → 形状 `(1, 64)`：
   ```python
   obs = np.concatenate([obs, est_vel, context_vec], axis=1)
   ```

### 3.2 Actor 推理步骤

1. **（可选）Actor 输入 RMS 归一化**（若配置了 `obs_rms_mean`）：
   ```python
   if self.obs_rms_mean is not None:
       obs = (obs - self.obs_rms_mean[None, :]) / np.sqrt(self.obs_rms_var[None, :] + 1e-8)
   ```

2. **调用 Actor 网络**，输出 12 维关节位置指令：
   ```python
   raw_actions = self.policy(obs).ravel()
   ```

### 3.3 控制线程与串行设计

- **控制线程**：整个控制循环在 `DreamWaqCENetVelocityState.policy_thread_handler()`（`unitree_sim2x/fsm/go2_states/dreamwaq_cenet_velocity_state.py:66‑101`）中运行。
- **频率**：线程以 **50 Hz（20 ms 周期）** 执行，每一步顺序完成 CENet 与 Actor 的推理。
- **为什么串行而非并行**？
  1. **数据依赖**：Actor 的输入依赖于 CENet 的输出（`est_vel`、`context_vec`），必须等待 CENet 完成推理后才能拼接。
  2. **计算量适中**：两个 ONNX 模型均较小，单线程顺序执行即可满足 50 Hz 实时性要求。
  3. **简化同步**：串行设计避免了多线程/进程间的数据同步与竞态问题。

---

## 4. 观测展开与历史缓冲区

### 4.1 数据结构

- **`obs_history_buffer`**（在 `CtrlEnv._init_cenet_support()` 中初始化）
  - 形状：`(history_length, single_obs_dim) = (5, 45)`
  - 内存布局（按时间顺序排列）：
    ```
    [obs(t-4)]  # 第 0 帧（最旧）
    [obs(t-3)]  # 第 1 帧
    [obs(t-2)]  # 第 2 帧
    [obs(t-1)]  # 第 3 帧
    [obs(t)]    # 第 4 帧（最新）
    ```
  - 每行 `obs(*)` 是 45 维的当前帧观测（已按 CENet 专用 RMS 参数归一化）。

- **`obs_history_flat`**（在 `CtrlEnv.step()` 中生成）
  - 通过对 `obs_history_buffer` 调用 `.flatten()` 得到一维数组，再通过 `[np.newaxis, :]` 添加批次维度：
    ```python
    obs_history_flat = self.obs_history_buffer.flatten()[np.newaxis, :]  # shape (1, 225)
    ```
  - 展平后的内存顺序为 **时间优先**（time‑first）：
    ```
    [obs(t-4)[0], obs(t-4)[1], …, obs(t-4)[44],
     obs(t-3)[0], obs(t-3)[1], …, obs(t-3)[44],
     …,
     obs(t)[0],   obs(t)[1],   …, obs(t)[44]]
    ```
  - 共计 `5 × 45 = 225` 个元素。

### 4.2 为何要展平？

- **模型输入要求**：CENet 是一个全连接网络（或含卷积层），其第一层接受 **固定长度的特征向量（225 维）**。
- **ONNX 导出**：模型在导出时已固定输入维度，运行时必须提供对应形状的张量。
- **训练‑部署对齐**：
  - 训练时（DreamWaq WAQ 任务）：`get_observation_history()` 返回的形状为 `(num_envs, history_length * obs_dim)`，即已展平。
  - 部署时：为保持与训练完全相同的输入格式，必须将 `(5, 45)` 的缓冲区展平为 `(1, 225)`。

---


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
