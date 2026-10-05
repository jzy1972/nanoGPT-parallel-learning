import torch.distributed as dist

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
