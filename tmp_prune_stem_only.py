import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1,3,224,224)
mod0 = base[0]; mod1 = base[1]
old_out = mod0.conv.out_channels
idxs = sorted(torch.topk(mod0.conv.weight.abs().mean(dim=(1,2,3)), max(1, int(old_out*0.15)), largest=False).indices.cpu().tolist())

# prune the stem output and the next block's input dims only
print('before', mod0.conv.out_channels, mod1.conv.in_channels)
tp.prune_conv_out_channels(mod0.conv, idxs)
keep = [i for i in range(mod0.bn.num_features) if i not in set(idxs)]
mod0.bn.weight.data = mod0.bn.weight.data[keep]
mod0.bn.bias.data = mod0.bn.bias.data[keep]
mod0.bn.running_mean = mod0.bn.running_mean[keep].clone()
mod0.bn.running_var = mod0.bn.running_var[keep].clone()
mod0.bn.num_features = len(keep)
tp.prune_conv_in_channels(mod1.conv, idxs)
print('after', mod0.conv.out_channels, mod1.conv.in_channels, mod1.bn.num_features)
try:
    y = base(x)
    print('forward ok', y.shape)
    print('params', sum(p.numel() for p in base.parameters()))
except Exception:
    import traceback; traceback.print_exc()
