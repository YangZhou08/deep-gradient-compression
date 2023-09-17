import torch
from torch import Tensor
# from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc
from typing import List, Optional
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
__all__ = ['SGDHessian', 'sgdhessian']

                
                
                
        
class SGDHessian(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        
        

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad


    @_use_grad_for_differentiable
    def step(self, closure=None, hessian_indices=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                hessian_indices=hessian_indices)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        hessian_indices: List[Tensor] = None):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize,
         hessian_indices=hessian_indices)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool,
                       hessian_indices: List[Tensor] = None):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        if not hessian_indices == None:
            raise Exception("hessian_indices not implemented in sgd hessian!")
        else:
            param.add_(d_p, alpha=-lr)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool,
                      hessian_indices: List[Tensor] = None):

    if len(params) == 0:
        return

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
    # for ((device_params, device_grads, device_momentum_buffer_list), indices) in grouped_tensors.values():
    for device_params, device_grads, device_momentum_buffer_list, indices in grouped_tensors.values():
        device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(tuple(device_grads))  # type: ignore[assignment]

        if weight_decay != 0:
            device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
                            torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        # print("len(hessian_indices)", len(hessian_indices))
        # print("len(device_params)", len(device_params))
        # print("len(device_grads)", len(device_grads))
        # for hessian_index, hessian_updates in enumerate(hessian_indices):
        #     print("hessian_updates.shape", hessian_updates.shape)
        #     print("device_params[hessian_index].shape", device_params[hessian_index].shape)
        #     print("device_grads[hessian_index].shape", device_grads[hessian_index].shape)
        # idx_t = 0
        # idx_r = 0
        if not device_has_sparse_grad:
            if not hessian_indices == None and len(device_params[0].size()) >= 2:
                # print("start idx", idx_t)
                # idx_t += 1
                for hessian_index, hessian_updates in enumerate(hessian_indices):
                    
                    # print("hessian_index", hessian_index)
                    # print("hessian_updates.shape device_params[hessian_index].shape device_grads[hessian_index].shape", hessian_updates.shape, device_params[hessian_index].shape, device_grads[hessian_index].shape)
                    # print("device_grads[hessian_index].shape", device_grads[hessian_index].shape)
                    # hessian_updates = hessian_indices[hessian_index]
                    # hessian_updates = torch.ones(device_params[hessian_index].size()).reshape(device_params[hessian_index].size()).cuda().to(torch.bool)
                    # print("device_grads[hessian_index][hessian_updates].shape", device_grads[hessian_index].shape, device_grads[hessian_index][hessian_updates].shape, device_grads[hessian_index][hessian_updates])
                    # print("device_params[hessian_index][hessian_updates].shape", device_params[hessian_index].shape, device_params[hessian_index][hessian_updates].shape, device_params[hessian_index][hessian_updates])
                    # device_params[hessian_index][hessian_updates].add_(device_grads[hessian_index][hessian_updates], alpha=-lr)
                    device_params[hessian_index][hessian_updates] = torch.add(device_params[hessian_index][hessian_updates], device_grads[hessian_index][hessian_updates], alpha=-lr) #.reshape(device_params[hessian_index].size())
                    # device_params[hessian_index][hessian_updates].add_(device_grads[hessian_index][hessian_updates], alpha=-lr).reshape(device_params[hessian_index].size())
                    # torch.add(device_params[hessian_index][hessian_updates], device_grads[hessian_index][hessian_updates], alpha=-lr)
                # torch._foreach_add_(device_params[hessian_indices[i]], device_grads[hessian_indices[]], alpha=-lr)
            else:
                # print("rrrrrr start idx", idx_r)
                # idx_r += 1
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)