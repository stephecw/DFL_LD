import numpy as np
import torch
from torch import nn

class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, dropout=0.0):
        super(CustomMLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Add an activation except for the last layer
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def evaluate(self, eval_solver, dataloader_eval):
        with torch.no_grad():
            self.eval()
            all_regrets = []
            for z, c, x, _, _ in dataloader_eval:
                c_hat_np = self.forward(z).detach().cpu().numpy()   # shape [B,n]
                x_true_cpu = x.float().cpu().numpy()               # shape [B,n], torch Tensor
                c_true_cpu = c.float().cpu().numpy()  

                for i in range(z.size(0)):
                    eval_solver.setObj(c_hat_np[i]) 
                    x_hat, _ = eval_solver.solve()
                    num = (c_true_cpu[i] * (x_true_cpu[i] - x_hat)).sum()
                    den = max((c_true_cpu[i] * x_true_cpu[i]).sum(), 1e-6)
                    rel_regret = num / den
                    all_regrets.append(rel_regret)

        return float(np.mean(all_regrets)), float(np.std(all_regrets))
