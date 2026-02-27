# Go2 DreamWaQ 项目代码梳理

## 项目概述

本项目是针对 Unitree Go2 四足机器人的行走演示代码集合，主要目标是通过深度强化学习实现机器人在复杂地形（尤其是崎岖地形）上的稳健运动，并提供从仿真到真实机器人的部署方案。

项目包含四个主要组件，每个组件针对不同的使用场景和研究目标：

1. **dreamwaq** - 主要研究项目：实现 DreamWaQ 算法，专注于崎岖地形下的强化学习运动控制
2. **gym_basic** - 基础示例：IsaacGym 环境下的基础机器人操作和检查
3. **unitree_rl_gym** - 参考实现：平坦地形上的标准强化学习运动控制
4. **wtw** - 仿真到现实部署：基于 "Walk These Ways" 算法的实际机器人部署

## 各组件详细分析

### 1. dreamwaq（核心研究组件）

#### 目标
- 独立实现 DreamWaQ 算法（论文《DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning》）
- 在崎岖地形上训练稳健的四足机器人运动策略
- 验证 Context-aided Estimator Network (CENet) 的有效性

#### 关键特性
- **CENet（上下文辅助估计网络）**：核心创新组件，用于估计机器人速度并生成地形上下文表示
- **多任务配置**：支持基础模型、Oracle模型（带真实速度）、WAQ模型（DreamWaQ）和Est模型（估计网络）
- **地形多样性**：支持平滑斜坡、粗糙斜坡、上下楼梯、离散地形等多种地形
- **域随机化**：提高策略的泛化能力

#### 代码结构
```
dreamwaq/
├── legged_gym/                    # 环境定义和训练脚本
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── base/             # 基础环境类（LeggedRobot）
│   │   │   ├── go2/              # Go2 特定配置（go2_config.py）
│   │   │   └── __init__.py       # 环境注册
│   │   ├── scripts/
│   │   │   ├── train.py          # 训练主程序
│   │   │   ├── play.py           # 推理演示
│   │   │   └── mini_test.py      # 单代理测试
│   │   └── utils/
│   │       ├── task_registry.py  # 任务注册
│   │       └── terrain.py        # 地形生成
│   └── resources/robots/go2/     # Go2 URDF 和网格文件
└── rsl_rl/                       # 强化学习算法实现
    └── rsl_rl/
        ├── algorithms/ppo.py     # PPO 算法
        ├── modules/actor_critic.py
        ├── runners/on_policy_runner.py  # 训练循环
        └── vae/
            ├── cenet.py          # CENet 实现（核心）
            └── estnet.py         # 对比估计网络
```

#### CENet 实现要点
- **架构**：编码器-解码器结构的变分自编码器（VAE）
- **输入**：观察历史（5个时间步的观测，共225维）
- **输出**：估计速度（3维）和地形上下文向量（16维）
- **损失函数**：速度估计损失 + 重建损失 + KL散度损失
- **训练**：与PPO算法协同训练，使用专门的 rollout storage

#### 训练配置选项
| 任务名称 | 配置类 | 观测维度 | 特权观测 | 说明 |
|----------|--------|----------|-----------|------|
| `--task=go2_base` | Go2RoughBaseCfg | 45 | 无 | 基础配置，无速度信息 |
| `--task=go2_oracle` | Go2RoughOracleCfg | 48 | 190 | 包含真实速度和地形高度信息 |
| `--task=go2_waq` | Go2RoughWaqCfg | 45 | 190 | DreamWaQ 配置，使用CENet |
| `--task=go2_est` | Go2RoughEstCfg | 45 | 190 | 估计网络对比配置 |

### 2. gym_basic（基础示例）

#### 目标
- 提供 IsaacGym 环境下 Go2 机器人的基础操作示例
- 演示摄像头使用、关节检查和基本运动控制

#### 主要文件
- `go2_cam.py`：第一人称摄像头视角演示
- `go2_inspection.py`：关节运动范围检查和可视化
- `egocentric_cam/`：自我中心摄像头配置

#### 用途
- 初学者入门 IsaacGym 和 Unitree Go2
- 验证机器人模型加载和基本功能

### 3. unitree_rl_gym（参考实现）

#### 目标
- 提供 Unitree 机器人在平坦地形上的标准强化学习示例
- 支持多种 Unitree 机器人（Go2, H1, H1_2, G1）

#### 特点
- 基于 legged_gym 和 rsl_rl 的标准实现
- 简化配置，专注于平坦地形运动
- 提供完整的训练和推理流程

#### 使用方法
```bash
# 训练
python legged_gym/scripts/train.py --task=go2

# 推理
python legged_gym/scripts/play.py --task=go2
```

### 4. wtw（仿真到现实部署）

#### 目标
- 将训练好的策略部署到真实的 Unitree Go2 机器人
- 基于 "Walk These Ways" 算法，适配 Unitree SDK2

#### 核心组件
- **训练部分** (`go2_gym`, `go2_gym_learn`)：
  - 自定义的 Go2 环境配置
  - PPO 算法实现
  - 预训练模型提供

- **部署部分** (`go2_gym_deploy`)：
  - **LCM 通信**：轻量级通信和编组，用于 PC 与机器人间的数据传输
  - **Unitree SDK2 适配**：替换原有的 UDP 基础 SDK
  - **lcm_position_go2.cpp**：核心部署文件，连接策略和低层控制
  - **部署脚本**：`deploy_policy.py` 加载策略并发送控制命令

#### 部署流程
1. **环境准备**：安装 LCM、Unitree SDK2
2. **编译部署代码**：构建 `lcm_position_go2` 可执行文件
3. **网络连接**：通过以太网连接 PC 和 Go2 机器人
4. **通信测试**：验证 LCM 和 SDK2 的通信
5. **策略部署**：运行部署脚本，通过游戏手柄控制机器人

#### 支持的平台
- **PC 部署**：通过有线连接控制机器人
- **Jetson Orin 部署**：在机器人 onboard 计算机上运行（开发中）

## 项目整体架构

### 技术栈
- **仿真引擎**：NVIDIA IsaacGym Preview 4
- **深度学习框架**：PyTorch 1.10+ with CUDA
- **强化学习算法**：PPO (Proximal Policy Optimization)
- **机器人控制**：Unitree SDK2（新版）、LCM 通信
- **开发语言**：Python（训练/推理）、C++（部署）、CMake（构建）

### 训练到部署流程
```
dreamwaq/unitree_rl_gym 训练
        ↓
    策略模型（.pt文件）
        ↓
wtw 部署框架加载模型
        ↓
通过 LCM 发送控制命令
        ↓
Unitree SDK2 执行运动
        ↓
真实 Go2 机器人运动
```

### 核心算法创新点
1. **DreamWaQ 算法**：通过隐式地形想象提高崎岖地形适应性
2. **CENet 网络**：同时估计速度和地形上下文的双目标网络
3. **域随机化策略**：系统延迟、摩擦随机化、质量变化等
4. **多地形课程学习**：从简单到复杂的地形渐进训练

## 配置详解

### Go2 机器人配置（go2_config.py）
- **关节配置**：12个自由度（每腿3个关节）
- **控制参数**：PD 控制，刚度 20 N*m/rad，阻尼 0.5 N*m*s/rad
- **观测空间**：45-238维（取决于任务）
- **动作空间**：12维（关节目标角度）
- **奖励函数**：速度跟踪、能量效率、姿态稳定等多目标

### 训练超参数
- **环境数量**：4096个并行环境
- **时间步长**：0.005秒（200Hz）
- **课程学习**：速度命令逐渐增加
- **域随机化**：摩擦系数 [0.2, 1.25]，质量变化 [-1.0, 2.0] kg

## 使用指南

### 快速开始（以 dreamwaq 为例）
```bash
# 1. 安装依赖
pip install -e dreamwaq/rsl_rl
pip install -e dreamwaq/legged_gym

# 2. 训练 DreamWaQ 模型
cd dreamwaq/legged_gym
python scripts/train.py --task=go2_waq --headless

# 3. 推理演示
python scripts/play.py --task=go2_waq --load_run=<run_name> --checkpoint=<iter>
```

### Docker 支持
项目提供 Dockerfile，可创建包含所有依赖的完整环境：
```bash
docker build . -t dreamwaq/dreamwaq -f docker/Dockerfile
```

## 实验结果

根据项目文档，DreamWaQ 模型在以下方面表现优异：

1. **地形适应性**：在平滑斜坡、粗糙斜坡、楼梯等多种地形上稳定行走
2. **速度估计精度**：CENet 能准确估计机器人线速度
3. **运动自然度**：相比基础模型，运动更加自然流畅
4. **抗干扰能力**：通过域随机化训练的策略具有较好的泛化性

## 项目贡献点

1. **算法实现**：完整独立的 DreamWaQ 算法实现
2. **平台适配**：将先进算法适配到 Unitree Go2 平台
3. **部署框架**：提供完整的仿真到现实部署方案
4. **代码可读性**：清晰的代码结构和详细注释
5. **多配置支持**：支持从基础到高级的多种训练配置

## 后续工作方向

1. **实时部署优化**：提高部署代码的实时性和稳定性
2. **更多地形挑战**：增加更复杂的地形类型
3. **多机器人协同**：扩展至多机器人协作场景
4. **实际环境测试**：在更多真实世界环境中验证算法
5. **算法改进**：探索更高效的训练方法和网络结构

## 总结

本项目提供了一个完整的四足机器人强化学习研究框架，从算法实现（dreamwaq）到基础示例（gym_basic），从参考实现（unitree_rl_gym）到实际部署（wtw），覆盖了机器人学习控制的完整链条。特别地，DreamWaQ 算法的实现展示了如何在崎岖地形上通过隐式地形想象提高运动稳定性，而 wtw 组件则解决了仿真到现实的关键部署问题。

项目代码结构清晰，配置灵活，既适合学术研究也适合工程应用，是 Unitree Go2 机器人控制的重要参考资源。