from typing import final, override,Callable
import torch
from torch import nn
from torch.optim import Adam

def goodnes(x:torch.Tensor):
    return x.pow(2).mean(1)

@final
class FFlayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device:torch.device|None=None, dtype:type|None=None,activation=None,lr:float=0.03,threshold:float=2.) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation=activation if activation is not None else nn.ReLU()
        self.opt=Adam(self.parameters(),lr)
        self.threshold=threshold
        self.p:float=0.01
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized:torch.Tensor=(input/input.norm(2,1,keepdim=True)).nan_to_num(0)
        lin_res= super().forward(normalized)
        return self.activation(lin_res)
    def train_(self,x_pos:torch.Tensor,x_neg:torch.Tensor,epochs=100)->tuple[torch.Tensor,torch.Tensor]:
        for _ in range(epochs):

            g_pos=self.forward(x_pos).pow(2).mean(1)
            g_neg=self.forward(x_neg).pow(2).mean(1)

            loss_pos=-g_pos+self.threshold
            loss_neg=g_neg+self.threshold
            
            get_close:Callable[[torch.Tensor],torch.Tensor]=lambda x: torch.log(1+torch.exp(x))
            loss:torch.Tensor=get_close(torch.cat([loss_pos,loss_neg])).mean()
            
            self.opt.zero_grad()
            _=loss.backward()
            _=self.opt.step()

        return self.forward(x_pos).detach(),self.forward(x_neg).detach()

class FFcontainer(nn.Module):
    def __init__(self, layers,device:torch.device|None=None,activation=nn.ReLU,lr=0.0003) -> None:
        super().__init__()
        self.layers=0;
        for (i,o) in zip(layers[:-1],layers[1:]):
            layers.append(FFlayer(i,o,device=device,activation=activation(),lr=lr))

if __name__=='__main__':
    print("Testing the forward layer")
    x_xor=torch.tensor([
        [0.,0.],
        [0.,1.],
        [1.,0.],
        [1.,1.],
    ])
    y_xor=torch.tensor([
        [0.],
        [1.],
        [1.],
        [0.],
    ])

    l1=FFlayer(2+1,50,activation=nn.LeakyReLU())
    l2=FFlayer(50,50,activation=nn.LeakyReLU())
    l3=FFlayer(50,1,activation=nn.LeakyReLU())  # Changed output size to 1 for XOR

    x_pos=torch.cat([x_xor,y_xor],dim=1)
    x_neg=torch.cat([x_xor,1.-y_xor],dim=1)
    print("x_neg:")
    print(x_neg)

    for iter in range(100):
        h1_pos, h1_neg = l1.train_(x_pos, x_neg, epochs=1)
        
        h2_pos_input = l2(h1_pos)
        h2_neg_input = l2(h1_neg)
        h2_pos, h2_neg = l2.train_(h2_pos_input, h2_neg_input, epochs=1)
        
        h3_pos_input = l3(h2_pos)
        h3_neg_input = l3(h2_neg)
        _, _ = l3.train_(h3_pos_input, h3_neg_input, epochs=1)

    # Test the network
    final_output = l3(l2(l1(x_pos)))
    print("Final output after training:")
    print(final_output)
    print("Expected output:")
    print(y_xor)
