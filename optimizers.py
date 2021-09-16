import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.functional as F
import math

import numpy as np

def _adam(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         weight_decay: float,
         eps: float,
         lr_tensors,
         mask_set):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, (param, lr_tensor, mask) in enumerate(zip(params, lr_tensors, mask_set)):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr_tensor / bias_correction1 * mask.float()

        # value deve essere uno scalare, a noi non va bene
        # param.addcdiv_(exp_avg, denom, value=-step_size)
        param.add_(-step_size * exp_avg / denom)

class AdamLRMasked(optim.Adam):
    def __init__(self, params, lr_tensors, masks, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super().__init__(params, lr=np.inf, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.lr_tensors = lr_tensors
        self.masks = masks

    @torch.no_grad()
    def step(self, active_masks, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            mask_set = []

            for p, mask_variants in zip(group['params'], self.masks):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

                cumulative_mask = torch.zeros_like(p, dtype=torch.bool)
                for mask_variant, active in zip(mask_variants, active_masks):
                    if active:
                        cumulative_mask |= mask_variant

                mask_set.append(cumulative_mask)

            beta1, beta2 = group['betas']

            _adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['weight_decay'],
                   group['eps'],
                   self.lr_tensors,
                   mask_set
                   )
        return loss