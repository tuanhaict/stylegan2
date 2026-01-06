import torch
import torch.nn as nn
import torch.nn.functional as F
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

def patch_synthesis_layer(layer: torch.nn.Module):
    assert hasattr(layer, "weight")

    old_forward = layer.forward

    def forward_lora(self, x, w, noise_mode='random', fused_modconv=True, **kwargs):
        # ---- copy logic từ SynthesisLayer.forward ----
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode != 'none':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device)

        W = self.weight
        if hasattr(self, "lora"):
            W = self.lora.apply(W)

        x = modulated_conv2d(
            x=x,
            weight=W,
            styles=styles,
            noise=noise,
            up=self.up,
            down=self.down,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=self.flip_weight,
            fused_modconv=fused_modconv
        )

        x = self.bias_act(x)
        return x

    layer.forward = types.MethodType(forward_lora, layer)
def inject_lora(G, rank=8, alpha=1.0):
    count = 0

    for m in G.modules():
        if m.__class__.__name__ == "SynthesisLayer":
            # gắn LoRA
            m.lora = LoRAWeight(m.weight, rank, alpha).to(m.weight.device)
            m.weight.requires_grad_(False)

            # patch forward
            patch_synthesis_layer(m)
            count += 1

    print(f"[LoRA] Injected & patched {count} SynthesisLayers")
