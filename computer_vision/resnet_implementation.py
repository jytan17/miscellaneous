import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
    
    def __init__(self, in_c, out_c, stride=1, first=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=3, 
            stride=stride, 
            padding=1)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_c, out_c, 1, stride) if first else None
        self.relu = nn.ReLU()        
        
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        if self.conv3:
            return y + self.conv3(x)
        else:
            return y
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_c, out_c, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResidualLayer(in_c, out_c, 2, True))
        for i in range(1, num_layers):
            self.layers.append(ResidualLayer(out_c, out_c))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x


def resnet(in_channel, classes):
	model = nn.Sequential(
		# the base block is a little bit special compared the the rest of the network
		# it takes in the one channel and is the only layer to use max pool
		nn.Sequential(
		nn.Conv2d(in_channel, 64, 7, 2, 3),
		nn.BatchNorm2d(64),
		nn.ReLU(),
		nn.MaxPool2d(3, 2, 1)
		), 
		
		# the resnet blocks, each made of 2 layers
		# Note: the first layer of the resnet block is takes a stride of 2 
		# to reduce the size of the feature maps for subsequent layers
		ResidualBlock(64, 64, 2), 
		ResidualBlock(64, 128, 2), 
		ResidualBlock(128, 256, 2), 
		ResidualBlock(256, 512, 2),

		# the global average pool essentiall takes the averegre of each "pixel" for each feature map, 
		# this cuts down the useage of a fully connected layer, thus cutting the number of paremeters dramatically
		nn.AdaptiveAvgPool2d((1,1)),
		nn.Flatten(),
		nn.Linear(512, classes)
	)
	return model
