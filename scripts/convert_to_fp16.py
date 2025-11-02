import torch
import sys
from pathlib import Path

# Usage: python convert_to_fp16.py <input.pt> <output.pt>
if len(sys.argv) != 3:
    print("Usage: python convert_to_fp16.py <input.pt> <output.pt>")
    sys.exit(1)

inp = Path(sys.argv[1])
out = Path(sys.argv[2])

ckpt = torch.load(inp, map_location="cpu")

def to_half(obj):
    if isinstance(obj, torch.Tensor):
        # convert floating point tensors to half
        return obj.half() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: to_half(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_half(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_half(v) for v in obj)
    return obj

ckpt_half = to_half(ckpt)
torch.save(ckpt_half, out)
print(f"Saved FP16 checkpoint: {out} (original: {inp})")