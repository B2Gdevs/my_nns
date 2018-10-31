import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch.autograd import Variable

array = torch.randn(( 50,1,2)).unsqueeze(0)
print(type(array))

print(array.view(-1, array.shape[1]))
upsample = nn.ConvTranspose2d(50, 1, 3) # upsamples rows and cols by num-1 on each.
array = upsample(array)
print(array.view(-1, array.shape[1]))
print(array)
print(array.shape)