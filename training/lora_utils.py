from training.lora import LoRAConv2d
import torch.nn as nn

def inject_lora(G, rank=8, alpha=1.0):
    for name, module in G.synthesis.named_modules():
        if isinstance(module, nn.Conv2d) and "torgb" not in name.lower():
            parent = G.synthesis
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            lora_layer = LoRAConv2d(module, rank, alpha)
            lora_layer.A.requires_grad = True
            lora_layer.B.requires_grad = True
            setattr(parent, last, lora_layer)
