# SPDX-License-Identifier: Apache-2.0
# This file is part of Unity ML-Agents Toolkit, licensed under the Apache License 2.0.
# Modified by IRIDIA / Ilyes Gharbi, Université libre de Bruxelles, 2025.
# Modifications: Adapt the original poca algorithm to take state instead of observation at the critic level and dissociate the state to the sensor at the actor level

from typing import Any, Dict, List
import numpy as np
from mlagents.torch_utils import torch, default_device
import copy

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from mlagents_envs.timers import timed

from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps

from mlagents.trainers.torch_entities.utils import ModelUtils

from .obs_select import pick_state_obs_index
import logging
log = logging.getLogger(__name__)

EPSILON = 1e-7  # Small value to avoid divide by zero


class SlicedActor(torch.nn.Module):
    """
    Thin wrapper around the real actor (actor_core) that:
      • Accepts the FULL observation list in BehaviorSpec order,
      • Slices out the indices the actor was actually built for,
      • Forwards common calls (forward/get_action_and_stats/get_stats),
      • Proxies normalization utilities (update_normalization/copy_normalization),
      • Sanitizes kwargs during ONNX export so we don't pass unsupported args.
    """

    def __init__(self, actor_core: torch.nn.Module, actor_obs_indices: List[int]):
        super().__init__()
        self.actor_core = actor_core
        self.actor_obs_indices = actor_obs_indices
        self._printed_shapes = False

    # Delegate unknown attributes to the core (e.g., action_spec, decoders, etc.)
    def __getattr__(self, name):
        # Avoid recursion before actor_core is set in __dict__
        if name != "actor_core" and "actor_core" in self.__dict__:
            core = self.__dict__["actor_core"]
            if hasattr(core, name):
                return getattr(core, name)
        return super().__getattr__(name)

    def _normalize_obs_list(self, obs_list):
        # Accept a single Tensor (export path) or a list/tuple of Tensors (runtime)
        if isinstance(obs_list, torch.Tensor):
            return [obs_list]
        if isinstance(obs_list, tuple):
            return list(obs_list)
        return obs_list

    def _slice(self, obs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Slice obs when we get the FULL list. If we already get the ACTOR subset
        (e.g., during ONNX export when ModelSerializer uses actor-only specs),
        just return as-is.
        """
        idx = self.actor_obs_indices
        if idx is None:
            return obs_list

        # If exporter (or caller) is already passing only the actor obs,
        # len(obs_list) will equal len(idx) and global indices won't apply.
        try:
            if len(obs_list) == len(idx) and max(idx, default=-1) >= len(obs_list):
                return obs_list
        except Exception:
            pass

        # If obs_list is shorter than the largest requested index, it must already be sliced.
        if len(obs_list) <= max(idx, default=-1):
            return obs_list

        return [obs_list[i] for i in idx]

    # Some ML-Agents code may call forward() directly (e.g., during ONNX export).
    # Sanitize kwargs to avoid passing unsupported ones to SimpleActor.forward().
    def forward(self, obs_list: List[torch.Tensor], *args, **kwargs):
        sliced = self._slice(obs_list)
        if not self._printed_shapes:
            try:
                def shp(x):
                    try:
                        return tuple(x.shape)
                    except Exception:
                        return x
                log.warning(
                    f"[SwarmACB] SlicedActor.forward full={[shp(t) for t in obs_list]} "
                    f"-> sliced={[shp(t) for t in sliced]} idx={self.actor_obs_indices}"
                )
            except Exception:
                pass
            self._printed_shapes = True

        # Try full kwargs; if SimpleActor.forward() doesn't accept them, relax.
        try:
            return self.actor_core(sliced, *args, **kwargs)
        except TypeError:
            # Keep only the common ones; drop 'sequence_length' etc. if unsupported.
            allowed = {"masks", "memories"}
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            try:
                return self.actor_core(sliced, *args, **filtered)
            except TypeError:
                # Final fallback: just pass the inputs
                return self.actor_core(sliced)

    # ML-Agents uses these helpers during training/inference
    def get_action_and_stats(self, obs_list: List[torch.Tensor], **kwargs):
        return self.actor_core.get_action_and_stats(self._slice(obs_list), **kwargs)

    def get_stats(self, obs_list: List[torch.Tensor], *args, **kwargs):
        return self.actor_core.get_stats(self._slice(obs_list), *args, **kwargs)

    # Normalization utilities used by the trainer
    def update_normalization(self, buffer) -> None:
        if hasattr(self.actor_core, "update_normalization"):
            return self.actor_core.update_normalization(buffer)

    def copy_normalization(self, other_module) -> None:
        if hasattr(self.actor_core, "copy_normalization"):
            return self.actor_core.copy_normalization(other_module)

    # Proxy common attributes/methods so the wrapper behaves like the actor
    @property
    def memory_size(self) -> int:
        return self.actor_core.memory_size

    def to(self, device):
        self.actor_core.to(device)
        return self

    def parameters(self, recurse: bool = True):
        return self.actor_core.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.actor_core.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.actor_core.load_state_dict(*args, **kwargs)



class TorchPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous actions, as well as recurrent networks.
        """
        super().__init__(seed, behavior_spec, network_settings)
        self.global_step = GlobalSteps()

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        # -------- SwarmACB: detect 5D per-agent state and decide actor routing --------
        full_obs_specs = self.behavior_spec.observation_specs
        obs_shapes = [tuple(spec.shape) for spec in full_obs_specs]
        self._state_idx = pick_state_obs_index(obs_shapes)

        if self._state_idx >= 0 and len(obs_shapes) > 1:
            # Exclude the 5D state from the ACTOR (decentralized execution)
            self._actor_obs_indices = [i for i in range(len(obs_shapes)) if i != self._state_idx]
            actor_obs_specs = [full_obs_specs[i] for i in self._actor_obs_indices]
            log.warning(
                f"[SwarmACB] TorchPolicy routing: actor_idx={self._actor_obs_indices} "
                f"(exclude state idx={self._state_idx}); obs_shapes={obs_shapes}"
            )
        else:
            # Fallback: actor uses all obs (no 5D state found, or it's the only obs).
            self._actor_obs_indices = list(range(len(obs_shapes)))
            actor_obs_specs = full_obs_specs
            if self._state_idx < 0:
                log.warning(f"[SwarmACB] TorchPolicy: no 5D state found; actor uses all obs: {obs_shapes}")
            else:
                log.warning(f"[SwarmACB] TorchPolicy: only one obs present (idx={self._state_idx}); actor uses all obs: {obs_shapes}")

        # Build the REAL actor on the SUBSET specs
        actor_core = actor_cls(
            observation_specs=actor_obs_specs,
            network_settings=network_settings,
            action_spec=behavior_spec.action_spec,
            **actor_kwargs,
        )
        actor_core.to(default_device())

        # Wrap it so callers (including ONNX exporter) can pass FULL obs and we slice inside.
        self.actor = SlicedActor(actor_core, self._actor_obs_indices)

        # Save the m_size needed for export
        self._export_m_size = self.m_size
        # m_size needed for training is determined by network, not trainer settings
        self.m_size = self.actor.memory_size  # proxies to core

        # One-time print guards
        self._printed_eval_shapes = False
        self._printed_train_shapes = False

    @property
    def export_memory_size(self) -> int:
        """
        Returns the memory size of the exported ONNX policy. This only includes the memory
        of the Actor and not any auxiliary networks.
        """
        return self._export_m_size

    def _extract_masks(self, decision_requests: DecisionSteps) -> np.ndarray:
        mask = None
        if self.behavior_spec.action_spec.discrete_size > 0:
            num_discrete_flat = np.sum(self.behavior_spec.action_spec.discrete_branches)
            mask = torch.ones([len(decision_requests), num_discrete_flat])
            if decision_requests.action_mask is not None:
                mask = torch.as_tensor(1 - np.concatenate(decision_requests.action_mask, axis=1))
        return mask

    @timed
    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        We now pass the FULL obs list to the wrapper; it slices internally.
        """
        obs = decision_requests.obs
        masks = self._extract_masks(decision_requests)
        tensor_obs = [torch.as_tensor(np_ob) for np_ob in obs]

        if not self._printed_eval_shapes:
            try:
                log.warning(
                    f"[SwarmACB] TorchPolicy.evaluate full_obs_shapes={[tuple(t.shape) for t in tensor_obs]} "
                    f"(actor_idx={self._actor_obs_indices})"
                )
            except Exception:
                pass
            self._printed_eval_shapes = True

        memories = torch.as_tensor(self.retrieve_memories(global_agent_ids)).unsqueeze(0)
        with torch.no_grad():
            action, run_out, memories = self.actor.get_action_and_stats(
                tensor_obs, masks=masks, memories=memories
            )
        run_out["action"] = action.to_action_tuple()
        if "log_probs" in run_out:
            run_out["log_probs"] = run_out["log_probs"].to_log_probs_tuple()
        if "entropy" in run_out:
            run_out["entropy"] = ModelUtils.to_numpy(run_out["entropy"])
        if self.use_recurrent:
            run_out["memory_out"] = ModelUtils.to_numpy(memories).squeeze(0)
        return run_out

    # --- SwarmACB: wrapper for optimizer so training uses the same slicing
    def get_actor_stats(
        self,
        tensor_obs: List[torch.Tensor],
        actions,
        masks=None,
        memories=None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        if not self._printed_train_shapes:
            try:
                log.warning(
                    f"[SwarmACB] TorchPolicy.train full_obs_shapes={[tuple(t.shape) for t in tensor_obs]} "
                    f"(actor_idx={self._actor_obs_indices})"
                )
            except Exception:
                pass
            self._printed_train_shapes = True
        return self.actor.get_stats(
            tensor_obs, actions, masks=masks, memories=memories, sequence_length=sequence_length
        )

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]

        run_out = self.evaluate(decision_requests, global_agent_ids)
        self.save_memories(global_agent_ids, run_out.get("memory_out"))
        self.check_nan_action(run_out.get("action"))
        return ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    def get_current_step(self):
        return self.global_step.current_step

    def set_step(self, step: int) -> int:
        self.global_step.current_step = step
        return step

    def increment_step(self, n_steps):
        self.global_step.increment(n_steps)
        return self.get_current_step()

    def load_weights(self, values: List[np.ndarray]) -> None:
        self.actor.load_state_dict(values)

    def init_load_weights(self) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        return copy.deepcopy(self.actor.state_dict())

    def get_modules(self):
        return {"Policy": self.actor, "global_step": self.global_step}

    # --- SwarmACB: expose actor obs subset for exporter ---
    def get_actor_obs_indices(self) -> List[int]:
        return self._actor_obs_indices

    def get_actor_observation_specs(self):
        full_specs = self.behavior_spec.observation_specs
        return [full_specs[i] for i in self._actor_obs_indices]
