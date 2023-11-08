from collections import OrderedDict
import torch
import torch.nn as nn
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
torch.manual_seed(2109)


'''
Create our 50-layer model.
'''
layer_dict = OrderedDict()
for i in range(50):
    linear_layer = nn.Linear(10, 10 if i < 49 else 5)
    custom_weights = torch.full(linear_layer.weight.shape, 0.4, requires_grad=True)
    linear_layer.weight.data = custom_weights
    layer_dict['lin{}'.format(i+1)] = linear_layer
    layer_dict['act{}'.format(i+1)] = nn.ReLU() # previously sigmoid

deep_neural_net = nn.Sequential(layer_dict)

deep_X = torch.randn(50, 10)
deep_Y = torch.randn(50, 5)

deep_Y_hat = deep_neural_net(deep_X)

deep_loss = nn.L1Loss()
deep_loss(deep_Y_hat, deep_Y).backward()


max_grad_magnitude_per_layer = []
for name, layer in deep_neural_net.named_modules():   
    if isinstance(layer, nn.Linear):
        max_grad_magnitude_per_layer.append(torch.max(torch.abs(layer.weight.grad)))

plt.xlabel('Layers')
plt.ylabel('Max Abs Gradient')
plt.plot(max_grad_magnitude_per_layer)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
plt.show()

'''
Ways to mitigate exploding gradient

- Use gradient clipping
- Change the initialization of the weights
- Batch normalization
- Weight Regularization
'''
