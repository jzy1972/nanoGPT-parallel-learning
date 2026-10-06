import torch.distributed as dist
import torch

def get_stage_layer_range(n_layer,pp_size,pp_rank):
    assert 1<=pp_size<=n_layer
    assert 0<=pp_rank<pp_size

    start = n_layer * pp_rank // pp_size
    end = n_layer * (pp_rank + 1) // pp_size

    return start,end

def build_pp_dp_groups(pp_size):
    world_rank=dist.get_rank()
    world_size=dist.get_world_size()

    assert pp_size>0
    assert world_size%pp_size==0

    dp_size=world_size//pp_size

    pp_group=None
    dp_group=None

    #创建pp组
    for d in range(dp_size):
        ranks=[d*pp_size+p for p in range(pp_size)]
        group=dist.new_group(ranks=ranks)

        if world_rank in ranks:
            pp_group=group

    #创建dp组
    for p in range (pp_size):
        ranks=[d*pp_size+p for d in range(dp_size)]
        group=dist.new_group(ranks=ranks)

        if world_rank in ranks:
            dp_group=group


    return pp_group,dp_group

def build_tied_weight_group(pp_size):
    world_rank=dist.get_rank()
    world_size=dist.get_world_size()

    assert world_size%pp_size==0

    dp_size=world_size//pp_size

    if pp_size==1:
        return None

    tied_group=None

    for d in range(dp_size):
        first_rank=d*pp_size
        last_rank=first_rank+pp_size-1

        ranks=[first_rank,last_rank]

        group=dist.new_group(ranks=ranks)

        if world_rank in ranks:
            tied_group=group

    return tied_group

@torch.no_grad()
def sync_tied_weights(stage,tied_group,src_rank):
    if stage.pp_size==1:
        assert stage.transformer["wte"].weight is stage.lm_head.weight
        return 

    if not(stage.is_first or stage.is_last):
        return

    if stage.is_first:
        weight=stage.transformer["wte"].weight

    if stage.is_last:
        weight=stage.lm_head.weight

    dist.broadcast(weight.detach(),src=src_rank,group=tied_group)

def send_forward(tensor,dst_rank,pp_group):
    assert dst_rank is not None

    dist.send(
        tensor.detach().contiguous(),
        dst=dst_rank,
        group=pp_group
    )

def recv_forward(
    shape,
    dtype,
    device,
    src_rank,
    pp_group,
    requires_grad=True,
):
    assert src_rank is not None

    tensor = torch.empty(
        shape,
        dtype=dtype,
        device=device,
    )

    dist.recv(
        tensor=tensor,
        src=src_rank,
        group=pp_group,
    )

    tensor.requires_grad_(requires_grad)
    return tensor

def send_backward(grad_input,dst_rank,pp_group):
    assert dst_rank is not None

    dist.send(
        tensor=grad_input.detach().contiguous(),
        dst=dst_rank,
        group=pp_group,
    )

def recv_backward(stage_output,src_rank,pp_group):
    assert src_rank is not None

    grad_output=torch.empty_like(
        stage_output,
        memory_format=torch.contiguous_format,
    )

    dist.recv(
        tensor=grad_output,
        src=src_rank,
        group=pp_group,
    )

    return grad_output

@torch.no_grad()
def get_dp_grad_weight(stage, targets, parallel, device):
    stats = torch.empty(
        2,
        dtype=torch.float64,
        device=device,
    )

    if stage.is_last:
        assert targets is not None

        local_valid = (targets != -1).sum().to(
            device=device,
            dtype=torch.int64,
        )

        global_valid = local_valid.clone()

        dist.all_reduce(
            global_valid,
            op=dist.ReduceOp.SUM,
            group=parallel["dp_group"],
        )

        stats[0] = (
            parallel["dp_size"]
            * local_valid.to(torch.float64)
            / global_valid.clamp_min(1).to(torch.float64)
        )

        stats[1] = global_valid.to(torch.float64)

    dist.broadcast(
        stats,
        src=parallel["pp_last_rank"],
        group=parallel["pp_group"],
    )

    if stats[1].item() == 0:
        raise ValueError("整个全局 batch 没有有效标签")

    return stats[0].item()

@torch.no_grad()
def sync_tied_weight_gradients(stage, tied_group):
    if stage.pp_size == 1:
        assert (
            stage.transformer["wte"].weight
            is stage.lm_head.weight
        )
        return

    if not (stage.is_first or stage.is_last):
        return

    assert tied_group is not None

    if stage.is_first:
        weight = stage.transformer["wte"].weight
    else:
        weight = stage.lm_head.weight

    assert weight.grad is not None

    dist.all_reduce(
        weight.grad,
        op=dist.ReduceOp.SUM,
        group=tied_group,
    )

@torch.no_grad()
def sync_dp_gradients(
    stage,
    dp_group,
    dp_size,
    grad_weight=1.0,
):
    assert dp_size > 0
    assert dist.get_world_size(group=dp_group) == dp_size

    for name, param in stage.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            raise RuntimeError(f"参数 {name} 缺少梯度")

        grad = param.grad
        grad.mul_(grad_weight)

        if dp_size > 1:
            dist.all_reduce(
                grad,
                op=dist.ReduceOp.SUM,
                group=dp_group,
            )
            grad.div_(dp_size)

