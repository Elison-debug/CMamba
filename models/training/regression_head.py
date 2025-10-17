import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, hidden_dim, out_dim=2, pool="last"):
        super().__init__()
        self.pool=pool; self.fc=nn.Linear(hidden_dim,out_dim)
    def forward(self,y):
        if y.dim()==2: h=y
        else: h=y[:,-1,:] if self.pool=="last" else y.mean(dim=1)
        return self.fc(h)
