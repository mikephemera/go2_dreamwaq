# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Modified for DreamWaQ export utilities

import os
import copy
import torch
import warnings
from typing import Optional, Tuple


# Copy of export_policy_as_jit from helpers.py
def export_policy_as_jit(actor_critic, path):
    """
    Export policy as TorchScript (JIT) format.
    """
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


# Copy of PolicyExporterLSTM from helpers.py
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            f"hidden_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )
        self.register_buffer(
            f"cell_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


# -------------------------------------------------------------------
# ONNX export functions
# -------------------------------------------------------------------

def export_policy_as_onnx(
    actor_critic,
    path: str,
    example_inputs: Optional[torch.Tensor] = None,
    opset_version: int = 14,
    verbose: bool = False,
):
    """
    Export policy network (actor) as ONNX format.

    Args:
        actor_critic: ActorCritic instance (from rsl_rl)
        path: Directory to save ONNX file (will create 'policy.onnx' inside)
        example_inputs: Example input tensor for tracing.
                        If None, will create based on actor input dimension.
        opset_version: ONNX opset version (default 14)
        verbose: Whether to print export details
    """
    import torch.onnx

    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "policy.onnx")

    # Determine if model is recurrent (LSTM)
    is_recurrent = hasattr(actor_critic, "memory_a")

    if is_recurrent:
        # LSTM network - use similar wrapper as JIT export
        class OnnxExporterLSTM(torch.nn.Module):
            def __init__(self, actor_critic):
                super().__init__()
                self.actor = copy.deepcopy(actor_critic.actor)
                self.is_recurrent = actor_critic.is_recurrent
                self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
                self.memory.cpu()
                self.register_buffer(
                    "hidden_state",
                    torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
                )
                self.register_buffer(
                    "cell_state",
                    torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
                )

            def forward(self, x):
                # x shape: [batch_size, input_dim]
                out, (h, c) = self.memory(
                    x.unsqueeze(0),  # add sequence dimension
                    (self.hidden_state, self.cell_state)
                )
                self.hidden_state[:] = h
                self.cell_state[:] = c
                return self.actor(out.squeeze(0))

            @torch.jit.export
            def reset_memory(self):
                self.hidden_state[:] = 0.0
                self.cell_state[:] = 0.0

        exporter = OnnxExporterLSTM(actor_critic)
        exporter.to("cpu")
        exporter.eval()

        # Create example input if not provided
        if example_inputs is None:
            input_dim = exporter.actor[0].in_features
            example_inputs = torch.randn(1, input_dim)
        else:
            example_inputs = example_inputs.to("cpu")

        # Export ONNX
        torch.onnx.export(
            exporter,
            example_inputs,
            onnx_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=verbose,
        )

    else:
        # MLP network
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        model.eval()

        # Create example input if not provided
        if example_inputs is None:
            input_dim = model[0].in_features
            example_inputs = torch.randn(1, input_dim)
        else:
            example_inputs = example_inputs.to("cpu")

        # Export ONNX
        torch.onnx.export(
            model,
            example_inputs,
            onnx_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=verbose,
        )

    if verbose:
        print(f"Exported policy as ONNX to: {onnx_path}")

    return onnx_path


def export_cenet_as_onnx(
    cenet,
    path: str,
    obs_history_dim: int,
    opset_version: int = 14,
    verbose: bool = False,
):
    """
    Export CENet (Context‑Aided Estimator Network) as ONNX format.

    Args:
        cenet: CENet instance
        path: Directory to save ONNX file (will create 'cenet.onnx' inside)
        obs_history_dim: Dimension of observation history input
        opset_version: ONNX opset version
        verbose: Whether to print export details
    """
    import torch.onnx

    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "cenet.onnx")

    # Move to CPU for export (ONNX export typically works on CPU)
    cenet = copy.deepcopy(cenet).to("cpu")
    cenet.eval()

    # Create example input
    example_input = torch.randn(1, obs_history_dim)

    # CENet forward returns: (est_next_obs, est_vel, mu, logvar, context_vec)
    # We need to define a wrapper that returns the outputs we need for inference
    class CenetInferenceWrapper(torch.nn.Module):
        def __init__(self, cenet):
            super().__init__()
            self.cenet = cenet

        def forward(self, obs_history):
            # During inference we only need est_vel and context_vec
            _, est_vel, _, _, context_vec = self.cenet(obs_history)
            return est_vel, context_vec

    wrapper = CenetInferenceWrapper(cenet)

    # Export ONNX
    torch.onnx.export(
        wrapper,
        example_input,
        onnx_path,
        opset_version=opset_version,
        input_names=["obs_history"],
        output_names=["est_vel", "context_vec"],
        dynamic_axes={
            "obs_history": {0: "batch_size"},
            "est_vel": {0: "batch_size"},
            "context_vec": {0: "batch_size"}
        },
        verbose=verbose,
    )

    if verbose:
        print(f"Exported CENet as ONNX to: {onnx_path}")

    return onnx_path


def export_estnet_as_onnx(
    estnet,
    path: str,
    obs_history_dim: int,
    opset_version: int = 14,
    verbose: bool = False,
):
    """
    Export ESTNet (Estimator Network) as ONNX format.

    Args:
        estnet: ESTNet instance (simple MLP estimator)
        path: Directory to save ONNX file (will create 'estnet.onnx' inside)
        obs_history_dim: Dimension of observation history input
        opset_version: ONNX opset version
        verbose: Whether to print export details
    """
    import torch.onnx

    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "estnet.onnx")

    # Move to CPU for export
    estnet = copy.deepcopy(estnet).to("cpu")
    estnet.eval()

    # Create example input
    example_input = torch.randn(1, obs_history_dim)

    # Export ONNX
    torch.onnx.export(
        estnet,
        example_input,
        onnx_path,
        opset_version=opset_version,
        input_names=["obs_history"],
        output_names=["est_vel"],
        dynamic_axes={
            "obs_history": {0: "batch_size"},
            "est_vel": {0: "batch_size"}
        },
        verbose=verbose,
    )

    if verbose:
        print(f"Exported ESTNet as ONNX to: {onnx_path}")

    return onnx_path


def export_models(
    ppo_runner,
    env,
    export_dir: str,
    export_jit: bool = True,
    export_onnx: bool = True,
    export_cenet: bool = False,
    export_estnet: bool = False,
    opset_version: int = 14,
    verbose: bool = True,
) -> dict:
    """
    Main function to export trained models in multiple formats.

    Args:
        ppo_runner: PPO runner instance (from task_registry.make_alg_runner)
        env: Environment instance
        export_dir: Base directory where 'exported/policies/' will be created
        export_jit: Whether to export policy as TorchScript (JIT)
        export_onnx: Whether to export policy as ONNX
        export_cenet: Whether to export CENet as ONNX (only if task uses CENet)
        export_estnet: Whether to export ESTNet as ONNX (only if task uses ESTNet)
        opset_version: ONNX opset version
        verbose: Print progress messages

    Returns:
        Dictionary with paths to exported files
    """
    import os
    import torch

    # Ensure export directory exists
    policies_dir = os.path.join(export_dir, "policies")
    os.makedirs(policies_dir, exist_ok=True)

    results = {}

    # Get policy network
    policy = ppo_runner.alg.actor_critic

    # Create example input for policy ONNX export
    # Actor input dimension can be obtained from the first linear layer
    if hasattr(policy.actor, '__getitem__'):
        # Sequential model
        input_dim = policy.actor[0].in_features
    else:
        # Maybe it's a custom module; try to infer from first linear layer
        for module in policy.actor.modules():
            if isinstance(module, torch.nn.Linear):
                input_dim = module.in_features
                break
        else:
            # Fallback: use observation dimension from environment
            input_dim = env.num_obs
            if verbose:
                warnings.warn(f"Cannot infer actor input dimension, using env.num_obs={input_dim}")

    device = env.device
    example_input = torch.randn(1, input_dim).to(device)

    # Export JIT
    if export_jit:
        try:
            export_policy_as_jit(policy, policies_dir)
            results["jit_policy"] = os.path.join(policies_dir, "policy_1.pt")
            if verbose:
                print(f"Exported policy as JIT to: {results['jit_policy']}")
        except Exception as e:
            warnings.warn(f"Failed to export JIT: {e}")

    # Export ONNX policy
    if export_onnx:
        try:
            onnx_path = export_policy_as_onnx(
                policy,
                policies_dir,
                example_inputs=example_input,
                opset_version=opset_version,
                verbose=verbose,
            )
            results["onnx_policy"] = onnx_path
        except Exception as e:
            warnings.warn(f"Failed to export ONNX policy: {e}")

    # Export CENet if requested and available
    if export_cenet:
        try:
            cenet = ppo_runner.get_inference_cenet(device=device)
            if cenet is not None:
                # Determine observation history dimension
                # This depends on env.cfg.env.num_observation_history and observation dimension
                obs_dim = env.num_obs
                hist_len = env.cfg.env.num_observation_history
                obs_history_dim = obs_dim * hist_len

                cenet_path = export_cenet_as_onnx(
                    cenet,
                    policies_dir,
                    obs_history_dim=obs_history_dim,
                    opset_version=opset_version,
                    verbose=verbose,
                )
                results["onnx_cenet"] = cenet_path
            else:
                if verbose:
                    print("CENet not available (ppo_runner.get_inference_cenet returned None)")
        except Exception as e:
            warnings.warn(f"Failed to export CENet: {e}")

    # Export ESTNet if requested and available
    if export_estnet:
        try:
            estnet = ppo_runner.get_inference_estnet(device=device)
            if estnet is not None:
                # Similar dimension calculation as CENet
                obs_dim = env.num_obs
                hist_len = env.cfg.env.num_observation_history
                obs_history_dim = obs_dim * hist_len

                estnet_path = export_estnet_as_onnx(
                    estnet,
                    policies_dir,
                    obs_history_dim=obs_history_dim,
                    opset_version=opset_version,
                    verbose=verbose,
                )
                results["onnx_estnet"] = estnet_path
            else:
                if verbose:
                    print("ESTNet not available (ppo_runner.get_inference_estnet returned None)")
        except Exception as e:
            warnings.warn(f"Failed to export ESTNet: {e}")

    if verbose:
        print(f"All exports completed. Results: {results}")

    return results