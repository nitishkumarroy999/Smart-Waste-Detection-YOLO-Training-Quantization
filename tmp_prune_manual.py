import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1, 3, 224, 224)
ordered = []
for mod in base.modules():
    if isinstance(mod, Conv):
        ordered.append(mod)
print('ordered len', len(ordered))

# prune only the backbone Conv wrappers, keeping the final Classify untouched
for idx, mod in enumerate(ordered[:-1]):
    conv = mod.conv
    if conv.out_channels <= 8:
        continue
    prune_count = max(1, int(conv.out_channels * 0.2))
    importance = conv.weight.abs().mean(dim=(1,2,3))
    prune_idxs = torch.topk(importance, prune_count, largest=False).indices.cpu().tolist()
    prune_idxs = sorted(set(prune_idxs))
    old_out = conv.out_channels
    print('pruning', idx, type(mod).__name__, 'out', old_out, 'count', len(prune_idxs), 'idxs', prune_idxs[:10])
    tp.prune_conv_out_channels(conv, prune_idxs)
    if hasattr(mod, 'bn') and mod.bn is not None:
        tp.prune_batchnorm_out_channels(mod.bn, prune_idxs)

    # propagate to downstream layers that consume the same channel count
    for later in ordered[idx+1:]:
        later_conv = later.conv
        if hasattr(later_conv, 'in_channels') and later_conv.in_channels == old_out:
            tp.prune_conv_in_channels(later_conv, prune_idxs)
            if hasattr(later, 'bn') and later.bn is not None:
                tp.prune_batchnorm_in_channels(later.bn, prune_idxs)
    try:
        out = base(x)
        print('forward ok after', idx, out.shape)
    except Exception as e:
        print('forward failed after', idx, type(e).__name__, e)
        break
    print('current head conv in', ordered[-1].conv.in_channels)

print('final param count', sum(p.numel() for p in base.parameters()))
