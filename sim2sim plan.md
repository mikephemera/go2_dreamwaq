Plan: Sim‑to‑Sim Transfer of Trained DreamWaQ Models from IsaacGym to MuJoCo

Context

The user has trained a locomotion policy for the Unitree Go2 robot in IsaacGym using the DreamWaQ framework (tasks: go2_base,
go2_oracle, go2_waq, go2_est). They now wish to run the same trained policy in a MuJoCo simulation (sim2sim) for validation,
visualization, or further training in a different simulator.

The codebase currently contains:

- Full IsaacGym environment implementations (legged_robot.py) with custom VecEnv interface.
- Trained model checkpoints (.pt files) that store PyTorch state dicts of actor‑critic networks and, for waq/est tasks, CENet/EstNet
  estimators.
- MuJoCo XML model files for H1_2 and Go1 robots, but no MuJoCo environment for Go2.
- No existing pipeline for transferring policies between simulators.

Goal

Create a pipeline that:

1.  Loads a trained DreamWaQ checkpoint (any task variant).
2.  Instantiates a MuJoCo environment for Go2 that matches the observation/action space of the IsaacGym environment.
3.  Runs the policy in MuJoCo with real‑time rendering and recording.

Recommended Approach

We will implement a new MuJoCo environment class that replicates the LeggedRobot interface, reuse the existing checkpoint‑loading
machinery, and write a dedicated inference script. The approach is chosen because:

- The observation‑computation and control logic are tightly coupled with the robot configuration; re‑implementing it in MuJoCo ensures
  compatibility.
- The existing OnPolicyRunner classes already handle loading of actor‑critic weights, RMS normalizers, and CENet/EstNet estimators.
- This allows us to support all task variants (base, oracle, waq, est) with minimal changes to the core training code.

Implementation Steps

Phase 1: Create MuJoCo Model for Go2

1.  Convert URDF to MuJoCo XML

- Use mujoco.urdf.load (Python) or the urdf_to_mjcf command‑line tool to convert
  dreamwaq/legged_gym/resources/robots/go2/urdf/go2.urdf.
- Ensure joint names match those in go2_config.py (FL_hip_joint, RL_hip_joint, …).
- Add position actuators with stiffness/damping from cfg.control.
- Save as dreamwaq/legged_gym/resources/robots/go2/mujoco/go2.xml.

2.  Validate Model Dynamics

- Write a small test script that loads the XML, applies the default joint angles, and steps the simulation.
- Verify that the robot stands stably with the PD gains from the config.

Phase 2: Implement the MuJoCo Environment Class

File: dreamwaq/legged_gym/envs/go2/go2_mujoco.py

3.  Class Structure

- Create Go2MuJoCo that mimics the public interface of LeggedRobot:
  - **init**: parse config, load MuJoCo model/data, allocate PyTorch buffers.
  - step(actions): apply PD control, step MuJoCo simulation, update observation history.
  - reset() / reset_idx(): reset robot state and command generator.
  - compute_observations(): replicate the observation‑computation logic from legged_robot.py.
  - get_observations(), get_privileged_observations(), get_true_vel(), get_observation_history().

4.  Observation Computation

- Compute base linear velocity (in base frame) from mj_data.qvel.
- Compute projected gravity.
- Sample commands via \_resample_commands (same ranges as in config).
- Compute DOF positions (offset by default_dof_pos) and velocities.
- Apply observation scaling (obs_scales) and clipping.
- For waq/est tasks, remove base_lin_vel from the observation vector (as in IsaacGym).

5.  PD Control

- Scale action with action_scale and add to default_joint_angles.
- Compute torque: stiffness _ (target_pos - current_pos) - damping _ current_vel.
- Apply torque via MuJoCo’s position actuators.

6.  Observation History

- Maintain a circular buffer of the last len_obs_history raw observations (size 45).
- Update after each step; used by CENet/EstNet.

Phase 3: Build the Inference Script

File: dreamwaq/legged_gym/scripts/mujoco_play.py

7.  Argument Parsing

- Re‑use the same CLI arguments as play.py (--task, --load_run, --checkpoint, --headless).
- Add --mujoco_xml to optionally override the XML path.

8.  Load Configuration and Runner

- Use task_registry to get the environment config (Go2RoughWaqCfg, etc.) and training config.
- Instantiate Go2MuJoCo with the environment config.
- Create the appropriate runner (OnPolicyRunner, OnPolicyRunnerWAQ, OnPolicyRunnerEst) using the MuJoCo environment and training
  config.
- Call runner.load(path) to restore actor‑critic weights, RMS normalizers, and CENet/EstNet weights.

9.  Policy Inference

- For base/oracle tasks: obtain policy = runner.get_inference_policy().
- For waq tasks: obtain cenet = runner.get_inference_cenet(); call cenet.forward(obs_history) to get est_vel and context_vec.
- For est tasks: obtain estnet = runner.get_inference_estnet(); call estnet.forward(obs_history) for est_vel.
- Build actor observation: concatenate normalized observation, estimated velocity, and (for waq) context vector.

10. Simulation Loop

- Reset environment.
- Each step:
  a. Get normalized observation (apply RMS if cfg.obs_rms).
  b. Build actor observation as above.
  c. Pass to policy (or actor_critic.act_inference).
  d. Call env.step(action).
  e. Render with MuJoCo’s viewer (glfw/mjv_makeScene).
- Record video (optional) with imageio or OpenCV.

Phase 4: Integrate CENet/EstNet

11. History Buffer Management

- Maintain a circular buffer of raw observations (size 45 × len_obs_history).
- Normalize with the same RMS used during training (if cfg.obs_rms).

12. Forward Pass

- For waq: est_next_obs, est_vel, mu, logvar, context_vec = cenet.forward(obs_history) (ignore mu, logvar).
- For est: est_vel = estnet.forward(obs_history).
- True velocity input is not needed for inference; pass a dummy tensor.

Phase 5: Testing and Verification

13. Observation Consistency

- Run a short IsaacGym episode with play.py --headless and log observations.
- Replay the same actions in MuJoCo and compare observation vectors (allow small numerical differences).

14. Policy Output Comparison

- Feed logged IsaacGym observations through the loaded policy and compare actions produced by IsaacGym vs. MuJoCo.

15. Qualitative Evaluation

- Visually inspect gait quality in MuJoCo.
- Measure episode length and cumulative reward; should be similar to IsaacGym performance.

Critical Files to Modify/Create

┌─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
│ File Path │ Purpose │
├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ dreamwaq/legged_gym/envs/go2/go2_mujoco.py │ Core MuJoCo environment class. │
├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ dreamwaq/legged_gym/scripts/mujoco_play.py │ Main inference script. │
├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ dreamwaq/rsl_rl/runners/on_policy_runner.py │ May need minor changes to expose get_inference_cenet/get_inference_estnet (if not │
│ │ already available). │
├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ dreamwaq/legged_gym/utils/helpers.py │ Add helper for loading MuJoCo model from URDF/XML. │
├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ dreamwaq/legged_gym/envs/go2/go2_config.py │ Optionally add a sim subclass for MuJoCo‑specific parameters. │
└─────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

Potential Challenges and Mitigations

┌──────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
│ Challenge │ Mitigation │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Dynamics mismatch (MuJoCo vs. PhysX) │ Start with flat terrain; tune contact and solver parameters empirically. Use same PD │
│ │ gains as IsaacGym. │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Observation noise and system delay │ Disable cfg.noise.add_noise and cfg.domain_rand.system_delay for initial transfer. │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Height‑measurement ray casting │ Skip for flat terrain (measure_heights = False). For rough terrain, implement mj_ray │
│ │ queries. │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Missing MuJoCo model │ Convert URDF with urdf_to_mjcf; verify joint limits and inertia. │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ Real‑time rendering performance │ Use mjv_makeScene/mjv_render with a separate rendering thread if needed. │
├──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
│ CENet/EstNet requires true velocity for │ For inference, call forward directly and ignore storage/update steps. │
│ training │ │
└──────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

Verification Strategy

1.  Observation Consistency Check: Compare IsaacGym and MuJoCo observations for the same actions.
2.  Policy Output Comparison: Compare actions generated from identical observations.
3.  Qualitative Evaluation: Visual inspection of gait; measure episode length and reward.

Next Steps (After Plan Approval)

1.  Create the MuJoCo XML model for Go2.
2.  Implement skeleton of Go2MuJoCo with placeholder observation computation.
3.  Write mujoco_play.py for base‑task checkpoints.
4.  Add CENet/EstNet support for waq/est tasks.
5.  Run verification tests and adjust dynamics parameters as needed.
