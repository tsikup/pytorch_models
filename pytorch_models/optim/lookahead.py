from collections import defaultdict, OrderedDict

import torch
from torch.optim import Optimizer

# """ Lookahead Optimizer Wrapper.
# Implementation modified from: https://github.com/alphadl/lookahead.pytorch
# Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
#
# Hacked together by / Copyright 2020 Ross Wightman
# """
#
#
class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = optimizer.defaults
        self.defaults.update(defaults)
        # super(Lookahead, self).__init__(optimizer.param_groups, optimizer.defaults)

        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # assert id(self.param_groups) == id(self.optimizer.param_groups)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer (self.optimizer). This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict[
                "param_groups"
            ],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)
