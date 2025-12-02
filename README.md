# go2_walk

<p align="center">
  <img src="./asset/Go2_walk.gif" alt="Go2 Walking Demo" width="600" />
</p>

<p align="center">
  <strong>Demonstration Code for <a href="https://www.unitree.com/go2/">Unitree Go2</a> Quadruped Robot Walking</strong>
</p>

<p align="center">
  <a href="#part-0-gym_basic">gym_basic</a> •
  <a href="#part-1-unitree_rl_gym">unitree_rl_gym</a> •
  <a href="#part-2-dreamwaq">dreamwaq</a> •
  <a href="#part-3-wtw">wtw</a>
</p>

---

## Project Structure

```
go2_walk/
├── gym_basic/          # Basic IsaacGym examples
├── unitree_rl_gym/     # Flat terrain locomotion
├── dreamwaq/           # Rough terrain with state estimation
│   ├── legged_gym/
│   └── rsl_rl/
└── wtw/                # Sim-to-Real deployment
    ├── go2_gym/
    ├── go2_gym_learn/
    └── go2_gym_deploy/
```

---

## Part 0. `gym_basic`

> **Go2 in IsaacGym**

- IsaacGym ver.4 basic examples
- Camera and joint inspection demos

---

## Part 1. `unitree_rl_gym`

> **Flat Terrain Locomotion**

| | |
|---|---|
| Original | [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) |
| Based on | [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) |

---

## Part 2. `dreamwaq`

> **Rough Terrain Locomotion with State Estimation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| Author | [Jungyeon Lee (curieuxjy)](https://github.com/curieuxjy) |
| Paper | [DreamWaQ: Learning Robust Quadrupedal Locomotion](https://arxiv.org/abs/2301.10602) |
| Related | [Fall Recovery Task](https://arxiv.org/abs/2306.12712) |
| Training Log | [WandB Dashboard](https://wandb.ai/curieuxjy/dreamwaq) |

> ⚠️ **Unofficial Implementation** - This is an independent reproduction, not affiliated with the original authors.

---

## Part 3. `wtw`

> **Sim-to-Real Deployment** *(For Your Attention)*

| | |
|---|---|
| Paper | [Walk These Ways](https://arxiv.org/abs/2212.03238) |
| Original | [Teddy-Liao/walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2) |
| Based on | [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) |

---

## References

- [go2_omniverse](https://github.com/abizovnuralem/go2_omniverse)
