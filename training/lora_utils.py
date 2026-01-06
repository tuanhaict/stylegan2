from training.lora import LoRAConv2d
import torch.nn as nn

def inject_lora(G, rank=8, alpha=1.0):
    """Inject LoRA into Generator's synthesis network"""
    count = 0
    
    # Debug: in ra tất cả conv layers
    print("\n=== Scanning for Conv2d layers ===")
    for name, module in G.synthesis.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Found Conv2d: {name}, shape: {module.weight.shape}")
    
    # Inject LoRA
    print("\n=== Injecting LoRA ===")
    for name, module in list(G.synthesis.named_modules()):
        if isinstance(module, nn.Conv2d) and "torgb" not in name.lower():
            # Parse module path
            if '.' not in name:
                # Top-level module
                parent = G.synthesis
                attr_name = name
            else:
                # Nested module
                parts = name.split('.')
                parent = G.synthesis
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
            
            # Replace with LoRA
            lora_layer = LoRAConv2d(module, rank, alpha)
            setattr(parent, attr_name, lora_layer)
            count += 1
            print(f"Injected LoRA into: {name}")
    
    print(f"\n=== Total LoRA layers injected: {count} ===")
    return count
