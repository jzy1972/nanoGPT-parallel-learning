import torch
import torch.nn as nn

from model import Block, LayerNorm
from parallel_utils import get_stage_layer_range

import math

class GPTStage(nn.Module):
    def __init__(self,config,pp_size,pp_rank):
        super().__init__()

        self.config=config
        self.pp_size=pp_size
        self.pp_rank=pp_rank

        self.is_first=(pp_rank==0)
        self.is_last=(pp_rank==pp_size-1)

        self.layer_start,self.layer_end=get_stage_layer_range(config.n_layer,pp_size,pp_rank)

        self.transformer=nn.ModuleDict()

        if self.is_first:
            self.transformer["wte"]=nn.Embedding(config.vocab_size, config.n_embd)
            self.transformer["wpe"]=nn.Embedding(config.block_size,config.n_embd)
            self.transformer["drop"]=nn.Dropout(config.dropout)

        self.transformer["h"] = nn.ModuleDict({
          str(i): Block(config)
          for i in range(self.layer_start, self.layer_end)
        })

        if self.is_last:
            self.transformer["ln_f"]=LayerNorm(config.n_embd, bias=config.bias)
            self.lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if self.is_first and self.is_last:
            self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param,
                                mean=0.0,
                                std=0.02/math.sqrt(2*config.n_layer),
                                )

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0, std=0.02)

    def forward(self,x):
        if self.is_first:
            B, T=x.size()
            assert T<=self.config.block_size

            pos=torch.arange(T,dtype=torch.long,device=x.device)

            tok_emb = self.transformer["wte"](x)
            pos_emb = self.transformer["wpe"](pos)
            x=self.transformer.drop(tok_emb+pos_emb)

        for block in self.transformer["h"].values():
            x=block(x)

        if self.is_last:
            x=self.transformer["ln_f"](x)
            x = self.lm_head(x)

        return x

    def load_dense_state_dict(self,dense_state_dict):
        stage_state_dict={}

        for name, local_tensor in self.state_dict().items():
            if name not in dense_state_dict:
                raise KeyError(f"原版权重中缺少：{name}")

            dense_tensor=dense_state_dict[name]

            if dense_tensor.shape != local_tensor.shape:
                raise KeyError(f"{name}的形状不匹配",
                               f"原版为 {tuple(dense_tensor.shape)}",
                               f"当前 stage 为 {tuple(local_tensor.shape)}"
                )

            stage_state_dict[name]=dense_tensor

        return self.load_state_dict(stage_state_dict,strict=True)



