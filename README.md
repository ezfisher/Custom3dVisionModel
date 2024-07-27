# Custom3dVisionModel
Custom pytorch implementation of the 3D convolution layer for computer vision models.

Done by splitting up tensor of size (batch size, channels, depth, height, width) into 3 tensors of size (batch size, channels, depth, height), (batch size, channels, depth, width), and (batch size, channels, height, width) and performing independent 2d conv layers on each slice. The three different outputs are unsqueezed to make the right shape and added back together. The output size and number of parameters for the custom layer are the same as those for the traditional implementation

Time comparison:
Tested on CPU
Each layer tested on the same random input of size (4, 3, 224, 224, 224) for 100 trials.

Time to run (mean +/- std):

custom: 0.66 +/- 0.05 sec

pytorch: 1.42 +/- 0.08 sec
