import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils.ops import bias_act
from training.networks import modulated_conv2d
import types
class LoRAWeight(torch.nn.Module):
    def __init__(self, weight, rank=8, alpha=1.0):
        super().__init__()
        out_c, in_c, k, _ = weight.shape
        self.rank = rank
        self.scale = alpha / rank

        self.A = torch.nn.Parameter(torch.randn(rank, in_c * k * k) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(out_c, rank))

    def apply(self, W):
        delta = (self.B @ self.A).view(W.shape)
        return W + self.scale * delta

def patch_synthesis_layer(layer):
    # Lưu forward gốc
    if not hasattr(layer, '_original_forward'):
        layer._original_forward = layer.forward
    
    def forward_lora(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        
        styles = self.affine(w)
        
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device
            ) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
        
        W = self.weight
        if hasattr(self, "lora"):
            W = self.lora.apply(W)
        
        flip_weight = (self.up == 1)
        
        x = modulated_conv2d(
            x=x,
            weight=W,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv
        )
        
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp
        )
        return x
    
    # Đừng dùng types.MethodType, gán trực tiếp
    layer.forward = forward_lora.__get__(layer, layer.__class__)

def inject_lora(G, rank=8, alpha=1.0):
    for m in G.modules():
        if m.__class__.__name__ == "SynthesisLayer":
            m.lora = LoRAWeight(m.weight, rank, alpha).to(m.weight.device)
            m.weight.requires_grad_(False)
            patch_synthesis_layer(m)

