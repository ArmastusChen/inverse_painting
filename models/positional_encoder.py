
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F





# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * np.pi * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim





class PositionalEncoder(nn.Module):
    def __init__(self,in_features=42 ):
        super(PositionalEncoder, self).__init__()

        Embedder, _ = get_embedder(10)
        self.Embedder = Embedder

        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features=in_features, out_features=256)  # from 42 to 256 dimensions
        # Define the second fully connected layer
        self.fc2 = nn.Linear(in_features=256, out_features=512)  # from 256 to 512 dimensions
        # Define the third fully connected layer
        self.fc3 = nn.Linear(in_features=512, out_features=768)  # from 512 to 768 dimensions

    def forward(self, x):
        x = self.Embedder(x)
        # Apply the first fully connected layer and a ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer and a ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer
        x = self.fc3(x)
        # Note: Depending on the task, you might want to add an activation function here as well
        return x[:, None, :]
