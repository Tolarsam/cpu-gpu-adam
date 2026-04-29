import gc
import threading
import time
import torch
from torch.optim.optimizer import (
    Optimizer, 
    ParamsT, 
    _get_scalar_dtype, 
    _device_dtype_check_for_fused,
    _use_grad_for_differentiable,
)
from torch.optim.adam import adam
from torch import Tensor
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from torch import nn
import queue

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam as _DSCPUAdam
    from deepspeed.ops.op_builder import CPUAdamBuilder
except ImportError:
    _DSCPUAdam = None
    CPUAdamBuilder = None

class CPUAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        grad_acc_step: int = 1, 
        max_gpu_tail_params: int = 0,
        warm_step: int = 3,
        adamw_mode=True,
        bias_correction=True,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        decoupled_weight_decay: bool = False,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Support AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
            
        # init param info
        self.worker_param_info: Dict[nn.Parameter, Dict[str, Any]] = {} 
        self._hooks_attached = False
        self.grad_acc_step = grad_acc_step # 累积多少次再进行更新
        # CUDA streams
        # 为每个device与cpu之间构建stream
        self.d2h_stream: Dict[torch.device, torch.cuda.Stream] = {}
        self.h2d_stream: Dict[torch.device, torch.cuda.Stream] = {}
        self.opt_param_num = sum(len(group['params']) for group in self.param_groups)
        self.max_gpu_tail_params = max_gpu_tail_params if max_gpu_tail_params is not None else self.opt_param_num
        self.gpu_tail_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type == "cuda":
                    cpu_copy = p.detach().to("cpu", non_blocking=False).pin_memory()
                else:
                    cpu_copy = p.detach().pin_memory() if not p.is_pinned() else p.detach()
                device = p.device
                if device.type != 'cuda':
                    d2h_stream = None
                    h2d_stream = None
                else:
                    if device  not in self.d2h_stream:
                        self.d2h_stream[device] = torch.cuda.Stream(device=device)
                        self.h2d_stream[device] = torch.cuda.Stream(device=device)
                    d2h_stream = self.d2h_stream[device]
                    h2d_stream = self.h2d_stream[device]
                self.worker_param_info[p] = {
                    "cpu_data": cpu_copy,
                    "grad_cpu": None,
                    "cpu_synced": True,
                    "is_gpu": False,
                    "opt_order": -1,
                    "group": group,
                    "d2h_stream": d2h_stream,
                    "h2d_stream": h2d_stream
                }
            
            group['gpu_update_params'] = []
        
        self.param_opt_order = 0
        # 队列与线程
        self.micro_step = 0
        self.cpu_queue: queue.Queue = queue.Queue()
        self.gpu_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.cpu_worker = threading.Thread(target=self._cpu_worker_loop, daemon=True)
        self.gpu_worker = threading.Thread(target=self._gpu_worker_loop, daemon=True)
        self.cpu_worker.start()
        self.gpu_worker.start()

        self._cpu_backend_ready = False
        self.opt_id = 0
        self.ds_opt_adam = None

        self._lock = threading.Lock()

        self._attach_backward_hooks()

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adamw_mode = adamw_mode
        self.bias_correction = bias_correction
        self.warm_step = warm_step

        self.last_cpu_param_step_time = None
        self.last_cpu_queue_size = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _data_check_stream(self, param: nn.Parameter):
        info = self.worker_param_info[param]
        if param.device.type == "cuda":
            if info["d2h_stream"] is None:
                device = param.device 
                info["d2h_stream"] = torch.cuda.Stream(device=device)
                info["cpu_data"] = info["cpu_data"].pin_memory()
            if info["h2d_stream"] is None:
                device = param.device
                info["h2d_stream"] = torch.cuda.Stream(device=device)

    def _attach_backward_hooks(self):
        if self._hooks_attached:
            return

        def make_hook(param: nn.Parameter):
            def hook(_):
                grad = param.grad
                info = self.worker_param_info[param]
                is_gpu = info["is_gpu"]
                if info["opt_order"] == -1:
                    info["opt_order"] = self.param_opt_order
                    self.param_opt_order += 1
                if (self.micro_step+1) % self.grad_acc_step != 0:
                    # 累积梯度，暂不更新
                    # info["grad_cpu"] = grad.detach().cpu()
                    return None
                # print(f"Hook triggered for param with opt_order {info['opt_order']}, is_gpu: {is_gpu}, micro_step: {self.micro_step}")
                self.last_cpu_param_step_time = time.time()
                self.last_cpu_queue_size = self.cpu_queue.qsize()
                if is_gpu:
                    # GPU 参数：梯度已在 GPU，不拷贝；加入事件以保证计算流完成
                    if param.device.type == "cuda":
                        prod_event = torch.cuda.Event()
                        torch.cuda.current_stream().record_event(prod_event)
                        self.gpu_queue.put((param, prod_event))
                    else:
                        self.gpu_queue.put((param, None))
                else:
                    # CPU 参数：异步拷回
                    if param.device.type == "cuda":
                        self._data_check_stream(param)
                        prod_event = torch.cuda.Event()
                        torch.cuda.current_stream().record_event(prod_event)
                        with torch.cuda.stream(info["d2h_stream"]):
                            info["d2h_stream"].wait_event(prod_event)
                            grad_cpu = grad.detach().to("cpu", non_blocking=True).pin_memory()
                        finish_event = torch.cuda.Event()
                        info["d2h_stream"].record_event(finish_event)
                        self.cpu_queue.put((param, grad_cpu, finish_event))
                    else:
                        self.cpu_queue.put((param, grad.detach(), None))
                return None
            return hook

        for group in self.param_groups:
            for p in group["params"]:
                p.register_post_accumulate_grad_hook(make_hook(p))
        self._hooks_attached = True

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            if len(grads) > 0:
                adam(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group["amsgrad"],
                    has_complex=has_complex,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    decoupled_weight_decay=group["decoupled_weight_decay"],
                )
        # self.cpu_queue.join()
        # self.gpu_queue.join()
        self.wait_and_schedule()
        torch.cuda.synchronize()
        return loss
    

    def _cpu_worker_loop(self):

        while not self.stop_event.is_set():
            try:
                param, grad_cpu, event = self.cpu_queue.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.001)  # 让出时间片
                continue
            if event is not None:
                event.synchronize()

            info = self.worker_param_info[param]
            # 单参数 step（CPUAdam 支持单独传入）
            
            group = info['group']
            beta1, beta2 = group["betas"]
            self.cpu_step(
                param,
                info["cpu_data"], 
                grad_cpu,
                betas=(beta1, beta2),
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )
            # 回写到 param.data 
            if param.device.type == "cuda":
                # 使用 h2d_stream 进行异步回写
                with torch.cuda.stream(info["h2d_stream"]):
                    param.data.copy_(info["cpu_data"], non_blocking=True)
            info["grad_cpu"] = None 
            self.cpu_queue.task_done()

    def _gpu_worker_loop(self):
        while not self.stop_event.is_set():
            try:
                param, event = self.gpu_queue.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.001)  # 让出时间片
                continue
            if event is not None:
                event.synchronize()
            info = self.worker_param_info[param]
            if info['is_gpu'] and param.grad is not None:
                self.gpu_step(param, param.grad)
            self.gpu_queue.task_done()
    
    def _init_cpu_backend(self):
        if self._cpu_backend_ready:
            return
        if _DSCPUAdam is None or CPUAdamBuilder is None:
            raise RuntimeError("需要 CPUAdam: pip install deepspeed")
        self.ds_opt_adam = CPUAdamBuilder().load()
        self.ds_opt_adam.create_adam(self.opt_id,
                                     self.lr,
                                     self.beta1,
                                     self.beta2,
                                     self.eps,
                                     self.weight_decay,
                                     self.adamw_mode,
                                     True)
        self._cpu_backend_ready = True

    def _lazy_state(self, ori_p:torch.Tensor, p: torch.Tensor):
        if ori_p not in self.state:
            self.state[ori_p] = {}
        st = self.state[ori_p]
        if st:
            return st
        state = st
        group = self.worker_param_info[ori_p]["group"]
        if group["fused"]:
            _device_dtype_check_for_fused(p)
        # note(crcrpar): [special device hosting for step]
        # Deliberately host `step` on CPU if both capturable and fused are off.
        # This is because kernel launches are costly on CUDA and XLA.
        state["step"] = (
            torch.zeros(
                (),
                dtype=_get_scalar_dtype(is_fused=group["fused"]),
                device=p.device,
            )
            if group["capturable"] or group["fused"]
            else torch.tensor(0.0, dtype=_get_scalar_dtype())
        )
        # Exponential moving average of gradient values
        state["exp_avg"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
        # Exponential moving average of squared gradient values
        state["exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
        if group["amsgrad"]:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state["max_exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
        return st

    def _maybe_move_state(self, p: torch.Tensor, device: torch.device):
        if p not in self.state:
            return
        st = self.state[p]
        for key in st:
            if torch.is_tensor(st[key]) and st[key].device != device:
                st[key] = st[key].to(device)

    @torch.no_grad()
    def cpu_step(self,
                 ori_p: torch.Tensor,
                 p: torch.Tensor,
                 grad: torch.Tensor,
                 lr: Optional[float],
                 betas: Optional[Tuple[float, float]],
                 eps: Optional[float],
                 weight_decay: Optional[float],
                 step_override: Optional[int] = None,
                 ):
        """
        使用 DeepSpeed CPUAdam 更新 (参数与梯度需在 CPU)。
        """
        if grad is None:
            return
        if p.device.type != "cpu" or grad.device.type != "cpu":
            raise RuntimeError("cpu_step 需要参数与梯度均在 CPU")
        self._init_cpu_backend()
        if p.numel() == 0:
                return
        if grad.is_sparse:
            grad = grad.to_dense()
        with self._lock:
            st = self._lazy_state(ori_p, p)
            if step_override is None:
                st['step'] += 1
                step = st['step']
            else:
                step = step_override
            lr_ = lr
            b1 = betas[0] 
            b2 = betas[1] 
            eps_ = eps 
            wd = weight_decay
            if isinstance(step, torch.Tensor):
                step_val = int(step.item())
            else:
                step_val = int(step)
            self.ds_opt_adam.adam_update(self.opt_id,
                                         step_val,
                                         lr_,
                                         b1,
                                         b2,
                                         eps_,
                                         wd,
                                         self.bias_correction,
                                         p.data,
                                         grad.data,
                                         st['exp_avg'],
                                         st['exp_avg_sq'])
            
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["gpu_update_params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex
            
    @torch.no_grad()
    def gpu_step(self,
                 p: torch.Tensor,
                 grad: torch.Tensor,
                ):
        """
        GPU 手写 Adam/AdamW。
        若指定 device: 将 param / grad / state 迁移至该 device 后更新。
        若未指定 device: 使用 grad.device；若 param 不在该 device 自动迁移。
        """
        if grad is None:
            return
        dev =  p.device
        if dev.type != "cuda":
            raise RuntimeError("gpu_step 目标 device 必须是 CUDA")

        if grad.device != dev:
            grad = grad.to(dev)

        with self._lock:
            self._maybe_move_state(p, dev)
    
    def wait_and_schedule(self):
        self.gpu_queue.join()
        queue_size = self.last_cpu_queue_size
        start = time.time()
        self.cpu_queue.join()
        end = time.time()
        # print(f"CPU queue size:{queue_size}, time cpu:{end - self.last_cpu_param_step_time:.3f}s")
        if self.warm_step <= 0 and \
            queue_size > 0 and \
            end - self.last_cpu_param_step_time > 0.03 and \
            self.gpu_tail_params < self.max_gpu_tail_params:
            # 如果等待时间过长，增加 GPU 尾部参数数量
            self.gpu_tail_params += queue_size
            self.gpu_tail_params = min(self.gpu_tail_params, self.max_gpu_tail_params)
            print(f"Increase gpu_tail_params to {self.gpu_tail_params}, number of params:{self.opt_param_num} , time cpu:{end - start:.3f}s")
            num_p = self.opt_param_num
            for p, info in self.worker_param_info.items():
                if info['opt_order'] >  num_p - self.gpu_tail_params - 1:
                    info['is_gpu'] = True
                    info['cpu_data'] = None  # 释放 CPU 副本
        if self.warm_step > 0:
            self.warm_step -= 1

    def close(self):
        self.stop_event.set()
        self.cpu_worker.join()
        self.gpu_worker.join()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool, optional): instead of setting to zero, set the
                grads to None. This is more memory efficient, but may have
                different behavior in some cases. Default: False.
        """
        super().zero_grad(set_to_none=set_to_none)
        self.micro_step = 0
        # for p, info in self.worker_param_info.items():
        #     info['cur_acc'] = 0

    
    def _memory_cleanup(self):
        """
        显存整理：释放已无引用的 GPU tensor，清理缓存，降低下一步分配 OOM 风险。
        """
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        # 可选：触发 IPC 内存收集（某些多进程场景）
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
