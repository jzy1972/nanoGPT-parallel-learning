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
