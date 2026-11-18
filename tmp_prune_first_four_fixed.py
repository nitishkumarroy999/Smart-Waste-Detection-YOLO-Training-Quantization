import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1,3,224,224)
conv_layers = [mod for mod in base.modules() if isinstance(mod, Conv)]

def trim_bn_out(bn, idxs):
    keep = [i for i in range(bn.num_features) if i not in set(idxs)]
    bn.weight.data = bn.weight.data[keep]
    bn.bias.data = bn.bias.data[keep]
    bn.running_mean = bn.running_mean[keep].clone()
    bn.running_var = bn.running_var[keep].clone()
    bn.num_features = len(keep)

ratio = 0.08
for idx in range(min(4, len(conv_layers)-1)):
    mod = conv_layers[idx]
    next_mod = conv_layers[idx+1]
    old_out = mod.conv.out_channels
    prune_count = max(1, int(old_out * ratio))
    imp = mod.conv.weight.abs().mean(dim=(1,2,3))
    prune_idxs = sorted(torch.topk(imp, prune_count, largest=False).indices.cpu().tolist())
    print('pruning idx', idx, 'old_out', old_out, 'prune_idxs', prune_idxs)
    tp.prune_conv_out_channels(mod.conv, prune_idxs)
    trim_bn_out(mod.bn, prune_idxs)
    if next_mod.conv.in_channels == old_out:
        tp.prune_conv_in_channels(next_mod.conv, prune_idxs)

try:
    y = base(x)
    print('forward ok', y.shape)
    print('params', sum(p.numel() for p in base.parameters()))
except Exception as e:
    import traceback; traceback.print_exc()
