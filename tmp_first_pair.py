import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
mod0 = base[0]
mod1 = base[1]

def bn_keep_and_trim(bn, idxs):
    keep = [i for i in range(bn.num_features) if i not in set(idxs)]
    bn.weight.data = bn.weight.data[keep]
    bn.bias.data = bn.bias.data[keep]
    bn.running_mean = bn.running_mean[keep].clone()
    bn.running_var = bn.running_var[keep].clone()
    bn.num_features = len(keep)

old_out = mod0.conv.out_channels
idxs = sorted(torch.topk(mod0.conv.weight.abs().mean(dim=(1,2,3)), max(1, int(old_out*0.2)), largest=False).indices.cpu().tolist())
print('before', mod0.conv.out_channels, mod0.bn.num_features, mod0.bn.running_mean.shape, mod1.conv.in_channels, mod1.bn.num_features, mod1.bn.running_mean.shape)

tp.prune_conv_out_channels(mod0.conv, idxs)
bn_keep_and_trim(mod0.bn, idxs)
print('after mod0', mod0.conv.out_channels, mod0.bn.num_features, mod0.bn.running_mean.shape)

tp.prune_conv_in_channels(mod1.conv, idxs)
bn_keep_and_trim(mod1.bn, idxs)
print('after mod1', mod1.conv.in_channels, mod1.bn.num_features, mod1.bn.running_mean.shape)
try:
    y = mod0(torch.randn(1,3,224,224))
    print('mod0 ok', y.shape)
    z = mod1(y)
    print('mod1 ok', z.shape)
except Exception as e:
    import traceback; traceback.print_exc()
