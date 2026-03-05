# Go2 DreamWaQ 项目代码结构梳理

本文档详细梳理了 `go2_dreamwaq` 项目的代码结构，涵盖四个主要组件：`dreamwaq`（主项目）、`gym_basic`（基础示例）、`unitree_rl_gym`（参考实现）和 `wtw`（仿真到实机部署）。每个组件的目录结构、关键文件、依赖关系及用途均有说明。

## 项目概述

本仓库集合了多个用于 Unitree Go2 四足机器人的强化学习（RL）仿真与部署代码。核心目标是实现**崎岖地形下的鲁棒运动控制**，主要基于 **DreamWaQ** 算法（ICRA 2023 QRC 冠军方案）。此外，还包含了基础的 IsaacGym 示例、平坦地形运动参考实现以及仿真到实机的部署代码。

**主要组件**：
1. **`dreamwaq/`** – 主项目，实现了 DreamWaQ 算法，包含上下文辅助估计网络（CENet）用于地形感知的速度估计。
2. **`gym_basic/`** – 基础的 IsaacGym 示例，包含相机视角和关节检查演示。
3. **`unitree_rl_gym/`** – 基于 legged_gym 的平坦地形运动参考实现，支持 Go2、H1、H1_2、G1 等机器人。
4. **`wtw/`** – “Walk These Ways” 仿真到实机部署项目，适配 Unitree Go2 的 SDK2 接口。

## 整体目录结构

```
go2_walk/
├── dreamwaq/           # [Main] 崎岖地形 + 状态估计
│   ├── legged_gym/     # 环境与任务定义
│   └── rsl_rl/         # PPO 算法实现（含 CENet）
├── gym_basic/          # [Sub] 基础 IsaacGym 示例
├── unitree_rl_gym/     # [Ref] 平坦地形运动参考
└── wtw/                # [Ref] 仿真到实机部署
    ├── go2_gym/        # 训练环境
    ├── go2_gym_learn/  # 学习算法
    └── go2_gym_deploy/ # 部署代码（含 LCM 与 SDK2）
```

## 1. dreamwaq（主项目）

**目标**：在崎岖地形上实现鲁棒的四足运动，利用 **Context-aided Estimator Network (CENet)** 进行速度估计，减少对真实速度信息的依赖。

### 1.1 核心算法与模块

| 模块 | 说明 |
|------|------|
| **CENet** | 上下文辅助估计网络，输入为观测历史与特权信息，输出估计的线速度。 |
| **EstNet** | 对比模型组使用的估计器。 |
| **PPO** |  proximal policy optimization 算法，使用 GPU 加速。 |
| **OnPolicyRunnerWaq** | 针对 DreamWaQ 定制的训练循环，在 RL 主循环前加入 CENet 估计阶段。 |

### 1.2 目录结构详解

```
dreamwaq/
├── legged_gym/                    # 环境定义（基于 leggedrobotics/legged_gym）
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── __init__.py        # 环境注册，供 task_registry 调用
│   │   │   ├── a1/a1_config.py    # A1 平台配置类，继承自 legged_robot_config
│   │   │   ├── base/
│   │   │   │   ├── legged_robot.py          # 运动任务的基环境类（LeggedRobot）
│   │   │   │   └── legged_robot_config.py   # 配置类（LeggedRobotCfg, LeggedRobotCfgPPO）
│   │   ├── scripts/
│   │   │   ├── train.py           # 主训练脚本（支持 wandb 设置）
│   │   │   ├── play.py            # 多智能体地形行走推断
│   │   │   └── mini_test.py       # 单智能体近距离观察推断
│   │   └── utils/
│   │       ├── logger.py          # play.py 和 mini_test.py 中使用的 matplotlib 绘图
│   │       ├── task_registry.py   # 根据 envs/__init__.py 注册的信息连接环境与算法
│   │       └── terrain.py         # 地形生成类，被 LeggedRobot 引用
│   └── resources/robots/a1/       # 机器人平台资源（URDF 与网格）
│       ├── urdf/a1.urdf
│       └── meshes/*.dae
└── rsl_rl/                        # RL 算法库（基于 leggedrobotics/rsl_rl）
    └── rsl_rl/
        ├── algorithms/
        │   └── ppo.py             # PPO 算法实现，使用 actor_critic.py 中的 Actor/Critic 类
        ├── modules/
        │   └── actor_critic.py    # Actor 与 Critic 网络类
        ├── runners/
        │   └── on_policy_runner.py # 包含主要 RL 循环的 OnPolicyRunner 类
        │                            # 基础模型使用 OnPolicyRunner，DreamWaQ 使用 OnPolicyRunnerWaq，
        │                            # Estnet 模型使用 OnPolicyRunnerEst
        ├── utils/
        │   └── rms.py              # 用于 CENet 正态先验分布训练的 Running Mean Std 类
        └── vae/
            ├── cenet.py            # Context‑Aided Estimator Network (CENet) 类
            └── estnet.py           # 对比模型组使用的 Estimator 类
```

### 1.3 关键文件说明

| 文件路径 | 作用 |
|----------|------|
| `dreamwaq/rsl_rl/rsl_rl/vae/cenet.py` | CENet 核心实现，输入为观测历史与特权信息，输出估计速度。 |
| `dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py` | 所有四足机器人环境的基类，处理仿真步骤、观测、奖励等。 |
| `dreamwaq/legged_gym/legged_gym/scripts/train.py` | 启动训练，支持 `--task=a1_base`、`a1_oracle`、`a1_waq` 等任务选项。 |
| `dreamwaq/legged_gym/legged_gym/utils/task_registry.py` | 注册环境与算法，使训练脚本能够根据任务名加载对应的配置与环境。 |

### 1.4 训练任务选项

| 任务选项 (`--task=`) | 配置类 | Critic 观测维度 | Actor 观测维度 | 说明 |
|---------------------|--------|----------------|----------------|------|
| `a1_base`           | A1RoughBaseCfg | 45 | 45 | 观测中不含线速度 |
| `a1_oracle`         | A1RoughOracleCfg | 238 | 238 | 真实线速度 + 特权信息（地形高度、摩擦系数） |
| `a1_waq`            | A1RoughBaseCfg | 238 | 64 | 估计线速度 + 特权信息 / 观测历史（5 个时间步） |

## 2. gym_basic（基础示例）

**目标**：提供 IsaacGym 的基础使用示例，方便用户快速上手 Go2 模型的加载、相机设置与关节动画。

### 2.1 目录结构

```
gym_basic/
├── go2_cam.py          # 带 egocentric 相机视角的 Go2 关节动画演示
├── go2_inspection.py   # Go2 关节检查与状态打印
├── cube.urdf           # 示例 URDF（立方体）
└── ball.urdf           # 示例 URDF（球体）
```

### 2.2 关键文件说明

| 文件 | 功能 |
|------|------|
| `go2_cam.py` | 创建 Go2 机器人，并在其雷达刚性体上绑定一个 egocentric 相机；实现关节范围动画，并定期保存 RGB 与深度图像。 |
| `go2_inspection.py` | 加载 Go2 模型，打印其刚体、关节、自由度信息，并实时显示接触力与关节状态。 |

### 2.3 依赖资源
- 机器人 URDF 与网格文件位于 `../dreamwaq/legged_gym/resources/robots/go2/`。

## 3. unitree_rl_gym（参考实现）

**目标**：为 Unitree 机器人（Go2、H1、H1_2、G1）提供平坦地形运动的参考实现，基于 **leggedrobotics/legged_gym** 和 **rsl_rl**。

### 3.1 目录结构

```
unitree_rl_gym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/          # 基础环境类（与 dreamwaq 类似）
│   │   ├── go2/go2_config.py    # Go2 配置
│   │   ├── h1/h1_config.py      # H1 配置
│   │   ├── h1_2/h1_2_config.py  # H1_2 配置
│   │   └── g1/g1_config.py      # G1 配置
│   ├── scripts/
│   │   ├── train.py       # 训练脚本
│   │   └── play.py        # 播放脚本
│   └── utils/
│       ├── terrain.py     # 地形生成
│       ├── logger.py      # 日志记录
│       └── isaacgym_utils.py # IsaacGym 工具函数
└── setup.py               # 包安装配置
```

### 3.2 关键文件说明

| 文件 | 功能 |
|------|------|
| `legged_gym/envs/go2/go2_config.py` | 定义 Go2 机器人的训练配置（观测维度、奖励系数、环境参数等）。 |
| `legged_gym/scripts/train.py` | 启动训练，支持 `--task=go2` 等参数。 |
| `setup.py` | 包依赖声明：`isaacgym`, `rsl-rl`, `matplotlib`, `numpy==1.20`, `tensorboard`。 |

### 3.3 支持的机器人
- **Go2**、**H1**、**H1_2**、**G1**

## 4. wtw（Walk These Ways – 仿真到实机部署）

**目标**：将 “Walk These Ways” 算法适配到 Unitree Go2，使用 **Unitree SDK2** 进行实机部署，支持通过 LCM 进行通信。

### 4.1 目录结构

```
wtw/
├── go2_gym/                 # 训练环境（基于 walk-these-ways）
│   ├── envs/
│   │   ├── base/           # 基础任务、配置、课程学习
│   │   ├── go2/            # Go2 专用配置与环境
│   │   └── rewards/        # 奖励函数
│   └── utils/              # 地形、数学工具等
├── go2_gym_learn/          # 学习算法（未展开）
├── go2_gym_deploy/         # 部署相关
│   ├── unitree_sdk2_bin/   # SDK2 二进制与库
│   ├── lcm_types/          # LCM 消息定义
│   ├── scripts/
│   │   └── deploy_policy.py # 部署策略主脚本
│   └── build/              # 编译输出（lcm_position_go2 等）
└── setup.py                # 项目安装配置
```

### 4.2 关键文件说明

| 文件 | 功能 |
|------|------|
| `go2_gym/envs/go2/go2_config.py` | Go2 专用配置，**注意**：修改参数应在此文件而非基类配置中。 |
| `go2_gym_deploy/unitree_sdk2_bin/lcm_position_go2.cpp` | 核心部署文件，将 LCM 消息与 SDK2 命令桥接。 |
| `go2_gym_deploy/scripts/deploy_policy.py` | 加载训练好的策略，通过 LCM 发送控制命令到机器人。 |
| `go2_gym_deploy/utils/cheetah_state_estimator.py` | 状态估计与手柄映射逻辑（`get_command` 函数）。 |

### 4.3 部署流程概要
1. **安装 LCM**：克隆并编译 LCM 库。
2. **编译 SDK2**：进入 `unitree_sdk2_bin/library/unitree_sdk2`，执行 `./install.sh` 并编译。
3. **编译 lcm_position_go2**：在 `go2_gym_deploy/build` 中执行 `cmake .. && make -j`。
4. **连接机器人**：通过以太网连接 Go2（IP: `192.168.123.161`），运行 `./lcm_position_go2 eth0`（替换为实际网卡）。
5. **启动策略**：运行 `python deploy_policy.py`，按 R2 键启动控制器。

### 4.4 手柄映射
- **R2**：启动控制器
- **L2+B**：切换阻尼模式（紧急停止）
- 其他方向与步态切换详见 `cheetah_state_estimator.py` 中的 `get_command` 函数。

## 依赖关系总结

| 组件 | 核心依赖 |
|------|----------|
| **dreamwaq** | IsaacGym Preview 4, rsl_rl, legged_gym, wandb, tensorboard, opencv‑python |
| **gym_basic** | IsaacGym Preview 4, numpy, matplotlib, Pillow |
| **unitree_rl_gym** | IsaacGym Preview 4, rsl_rl, matplotlib, numpy==1.20, tensorboard |
| **wtw** | IsaacGym Preview 4, pytorch 1.10 (CUDA 11.3), LCM, Unitree SDK2 |

## 快速启动指南

### 5.1 通用前提
- 安装 **NVIDIA IsaacGym Preview 4**（需从官方网站下载）。
- 确保显卡驱动支持 CUDA 12.1+。

### 5.2 运行 dreamwaq（示例）
```bash
cd dreamwaq
pip install -e ./rsl_rl
pip install -e ./legged_gym
pip install tensorboard wandb opencv-python
python legged_gym/scripts/train.py --task=a1_waq --headless
```

### 5.3 运行 gym_basic（示例）
```bash
cd gym_basic
python go2_cam.py
```

### 5.4 运行 unitree_rl_gym（示例）
```bash
cd unitree_rl_gym
pip install -e .
python legged_gym/scripts/train.py --task=go2 --headless
```

### 5.5 运行 wtw 部署（示例）
```bash
cd wtw
pip install -e .
# 编译部署代码（见上文）
cd go2_gym_deploy/build
sudo ./lcm_position_go2 eth0
# 另起终端
cd go2_gym_deploy/scripts
python deploy_policy.py
```

## 总结

本仓库通过四个独立又互补的组件，覆盖了从基础仿真、崎岖地形 RL 训练、平坦地形参考实现到仿真‑实机部署的全流程。**`dreamwaq`** 作为核心，展示了如何利用 CENet 提升在未知地形中的运动鲁棒性；**`gym_basic`** 为初学者提供了 IsaacGym 的入门示例；**`unitree_rl_gym`** 和 **`wtw`** 则为实际机器人部署提供了可参考的代码框架。

**注**：各组件均可独立使用，但共享部分资源（如机器人 URDF 文件位于 `dreamwaq/legged_gym/resources/`）。用户可根据需求选择相应的目录进行开发与实验。

---
*文档生成日期：2026‑03‑05*
*基于项目 README 与代码文件分析整理*