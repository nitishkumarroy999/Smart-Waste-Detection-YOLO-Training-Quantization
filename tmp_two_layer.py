import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1,3,224,224)
mod0 = base[0]
mod1 = base[1]

old_out = mod0.conv.out_channels
idxs = sorted(torch.topk(mod0.conv.weight.abs().mean(dim=(1,2,3)), max(1, int(old_out*0.2)), largest=False).indices.cpu().tolist())
print('before', old_out, mod1.conv.in_channels, mod1.bn.num_features)

# prune mod0 out, mod1 in and bn
print('prune mod0')
tp.prune_conv_out_channels(mod0.conv, idxs)
tp.prune_batchnorm_out_channels(mod0.bn, idxs)
print('after mod0', mod0.conv.out_channels, mod0.bn.num_features)

print('prune mod1')
tp.prune_conv_in_channels(mod1.conv, idxs)
tp.prune_batchnorm_in_channels(mod1.bn, idxs)
print('after mod1', mod1.conv.in_channels, mod1.bn.num_features)

try:
    y = mod0(x)
    print('mod0 ok', y.shape)
    z = mod1(y)
    print('mod1 ok', z.shape)
except Exception as e:
    import traceback; traceback.print_exc()
