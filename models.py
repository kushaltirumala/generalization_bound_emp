import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy
import math


class VGGnet(nn.Module):
    def __init__(self, num_classes_to_predict):
        super(VGGnet, self).__init__()
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = make_layers(cfg, 3)
        self.classifier = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes_to_predict))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_classes_to_predict):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes_to_predict)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, n_units, init_scale=1.0):
        super(MLP, self).__init__()

        self._n_units = copy.copy(n_units)
        self._layers = []
        for i in range(1, len(n_units)):
            layer = nn.Linear(n_units[i - 1], n_units[i], bias=False)
            variance = math.sqrt(2.0 / (n_units[i - 1] + n_units[i]))
            layer.weight.data.normal_(0.0, init_scale * variance)
            self._layers.append(layer)

            name = 'fc%d' % i
            if i == len(n_units) - 1:
                name = 'fc'  # the prediction layer is just called fc
            self.add_module(name, layer)

    def forward(self, x):
        x = x.view(-1, self._n_units[0])
        out = self._layers[0](x)
        # temp_sum = torch.sum(out)
        # out = out / temp_sum
        for layer in self._layers[1:]:
            out = F.relu(out)
            out = layer(out)
            # temp_sum = torch.sum(out)
            # out = out/temp_sum

        return out

# class SimpleMLP(nn.Module):
#     def __init__(self, n_units, init_scale=1.0):
#         super(MLP, self).__init__()
#
#         self._n_units = copy.copy(n_units)
#         self._layers = []
#         for i in range(1, len(n_units)):
#             layer = nn.Linear(n_units[i - 1], n_units[i], bias=False)
#             variance = math.sqrt(2.0 / (n_units[i - 1] + n_units[i]))
#             layer.weight.data.normal_(0.0, init_scale * variance)
#             self._layers.append(layer)
#
#             name = 'fc%d' % i
#             if i == len(n_units) - 1:
#                 name = 'fc'  # the prediction layer is just called fc
#             self.add_module(name, layer)
#
#     def forward(self, x):
#         x = x.view(-1, self._n_units[0])
#         out = self._layers[0](x)
#         # temp_sum = torch.sum(out)
#         # out = out / temp_sum
#         for layer in self._layers[1:]:
#             out = F.relu(out)
#             out = layer(out)
#             # temp_sum = torch.sum(out)
#             # out = out/temp_sum
#
#         return out

