from typing import final, override,Callable
import torch
from torch import nn
from torch.optim import AdamW

@final
class FFlayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device:torch.device|None=None, dtype:type|None=None,activation:nn.Module|None=None,lr:float=0.03,threshold:float=2.) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation=activation if activation is not None else nn.ReLU()
        self.opt=AdamW(self.parameters(),lr)
        self.threshold=threshold
        self.p:float=0.01
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized:torch.Tensor=input/torch.tensor(input.norm(2,1,keepdim=True))
        lin_res= super().forward(normalized)
        return torch.tensor(self.activation(lin_res))
    def train(self,x_pos:torch.Tensor,x_neg:torch.Tensor,epochs=100)->tuple[torch.Tensor,torch.Tensor]:
        for _ in range(epochs):
            g_pos=self.forward(x_pos).pow(2).mean(1)
            g_neg=self.forward(x_neg).pow(2).mean(1)

            loss_pos=-g_pos+self.threshold
            loss_neg=g_neg+self.threshold
            
            get_close:Callable[[torch.Tensor],torch.Tensor]=lambda x: torch.log(1+torch.exp(x))+self.p*x

            loss:torch.Tensor=get_close(torch.cat([loss_pos,loss_neg])).mean()
            
            self.opt.zero_grad()
            _=loss.backward()
            _=self.opt.step()

        return self.forward(x_pos).detach(),self.forward(x_neg).detach()
