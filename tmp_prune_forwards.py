import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1, 3, 224, 224)

# Flatten relevant Conv wrappers in backbone order, skipping the final head.
backbone_conv_layers = []
for mod in base.modules():
    if isinstance(mod, Conv):
        backbone_conv_layers.append(mod)
print('backbone conv count', len(backbone_conv_layers), 'head last type', type(backbone_conv_layers[-1]).__name__)

seen = set()
for i, mod in enumerate(backbone_conv_layers[:-1]):
    if id(mod) in seen:
        continue
    seen.add(id(mod))
    conv = mod.conv
    bn = getattr(mod, 'bn', None)
    if conv.out_channels <= 8:
        continue
    prune_count = max(1, int(conv.out_channels * 0.2))
    importances = conv.weight.abs().mean(dim=(1, 2, 3))
    prune_idxs = sorted(torch.topk(importances, prune_count, largest=False).indices.cpu().tolist())
    old_out = conv.out_channels
    print('pruning', i, type(mod).__name__, 'old_out', old_out, 'idxs', prune_idxs)

    # prune the current Conv + BN output channels
    tp.prune_conv_out_channels(conv, prune_idxs)
    if bn is not None:
        keep = [j for j in range(bn.num_features) if j not in set(prune_idxs)]
        bn.weight.data = bn.weight.data[keep]
        bn.bias.data = bn.bias.data[keep]
        bn.running_mean = bn.running_mean[keep]
        bn.running_var = bn.running_var[keep]
        bn.num_features = len(keep)

    # prune any direct consumer with matching input channels
    for later in backbone_conv_layers[i+1:]:
        later_conv = later.conv
        if later_conv.in_channels == old_out:
            tp.prune_conv_in_channels(later_conv, prune_idxs)
            later_bn = getattr(later, 'bn', None)
            if later_bn is not None:
                keep = [j for j in range(later_bn.num_features) if j not in set(prune_idxs)]
                later_bn.weight.data = later_bn.weight.data[keep]
                later_bn.bias.data = later_bn.bias.data[keep]
                later_bn.running_mean = later_bn.running_mean[keep]
                later_bn.running_var = later_bn.running_var[keep]
                later_bn.num_features = len(keep)
            break

try:
    y = base(x)
    print('forward ok', y.shape)
    print('params', sum(p.numel() for p in base.parameters()))
except Exception as e:
    import traceback; traceback.print_exc()
