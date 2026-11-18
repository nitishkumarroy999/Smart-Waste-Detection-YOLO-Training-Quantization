import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import torch_pruning as tp
from torch_pruning import function

class UltralyticsConvPruner(function.BasePruningFunc):
    TARGET_MODULES = Conv

    def prune_out_channels(self, layer, idxs):
        layer.conv = tp.prune_conv_out_channels(layer.conv, idxs)
        if hasattr(layer, 'bn') and layer.bn is not None:
            layer.bn = tp.prune_batchnorm_out_channels(layer.bn, idxs)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.conv.out_channels if hasattr(layer, 'conv') and layer.conv is not None else None

    def get_in_channels(self, layer):
        return layer.conv.in_channels if hasattr(layer, 'conv') and layer.conv is not None else None

m = YOLO(r'C:\Users\nitis\YOLO26\runs\classify\garbage_trashnetpp_13class_holdout\weights\best.pt')
base = m.model.model
x = torch.randn(1, 3, 224, 224)
customized = {Conv: UltralyticsConvPruner()}
pruner = tp.pruner.MagnitudePruner(
    base,
    x,
    importance=tp.importance.MagnitudeImportance(p=2),
    pruning_ratio=0.2,
    ignored_layers=[base[-1]],
    customized_pruners=customized,
    root_module_types=[Conv, nn.Conv2d],
)
print('registered', list(pruner.DG.REGISTERED_PRUNERS.keys())[:10])
print('conv count', sum(isinstance(m, (Conv, nn.Conv2d)) for m in base.modules()))
print('conv modules', [type(m).__name__ for m in base.modules() if isinstance(m, (Conv, nn.Conv2d))][:10])
for m in base.modules():
    if isinstance(m, (Conv, nn.Conv2d)):
        print('module', type(m), 'pruner', pruner.DG.get_pruner_of_module(m))
        break
print('module2node sample', list(pruner.DG.module2node.keys())[:10])
print('node count', len(pruner.DG.module2node))
