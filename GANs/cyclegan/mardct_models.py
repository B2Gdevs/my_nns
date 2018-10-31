import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # if the name has "Conv" in it # -1 from find() means it wasn't found as a substring
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # this initializes the weights to a normal distribution with a mean of 0 and std of 0.02
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # this initializes the weights to a normal distribution with a mean of 1 and std of 0.02
        torch.nn.init.constant_(m.bias.data, 0.0)  # this initializes the biases to 0

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1), # pads the matrix by reflecting it by 1 element in all directions.
                        nn.Conv2d(in_features, in_features, 3), # 3 in this case is the kernel size which
                        nn.InstanceNorm2d(in_features), # does instance normalization to each IMAGE rather than a batch of images
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x) # clearly a skip network aka ResBlock inputs are added with the features that were convolved.
                                      # This explains why the "in_features" are remaining the same, because how would the one-to-one
                                      # addition work otherwise here.

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3), # reflects the border by 3 elements from the original matrix # means 6 cols and 6 rows have been added to the matrix
                                           # this means if it were a 512x512 image it would now be 518x518
                    nn.Conv2d(in_channels, 64, 7), # There are 3 "filters" in an RGB image # this initializes and outputs 64 filters
                    nn.InstanceNorm2d(64), # again instance normalization is a per image normalization.
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64 # yeah obviously from the previous model InstanceNorm2d 5 lines up 
        out_features = in_features*2 # so now 128 filters wanted
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), # 1st loop: given 64, output 128, 2nd loop: 128, output 256
                                                                                      # size_of_feature_map =  ((X - (filter_size)) + 2(padding) / (stride)) + 1
                                                                                      # so size_of_feature_map = ((unknown-3)+(2(1)/2)) + 1 = UNKOWN
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features # 1st loop: input is now 128, 2nd loop: 256
            out_features = in_features*2 # 1st loop: output is now 256, 2nd loop: 512

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)] # so the input will be 256ximage_sizeximage_size, output will be the same
                                                  # that is how the residual blocks work

        # Upsampling
        out_features = in_features//2 # so inputs are 256, output is now 128
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), # 1st: in 256, out 128, 2nd: in 128, out 64
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features # 1st: 128, 2nd
            out_features = in_features//2 # 1st: 64, 2nd

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)