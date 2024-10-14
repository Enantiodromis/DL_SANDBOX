###########
# IMPORTS #
###########
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
from tqdm import tqdm
import torch

torch.manual_seed(42)

###########################################################
# ------------- SECTION 3 : CONTINIOUS XOR -------------  #
###########################################################
# It is possible to manually build a neural network in PyTorch by specifying all our parameters (weights and biases)
# manually by initialising multiple Tensors, calculate the gradients and then adjust the parameters. But this would
# become cumbersome quickly especially when there are a lot of parameters.
#
# PyTroch has a package called torch.nn that makes building neural networks more convenient. We introduce the package
# using a binary classification use-case using XOR. 
#
# BINARY CLASSIFICATION DEFINITION:
# Given two binary inputs x1 and x2, the label to predict is 1 if either x1 or x2 is 1 while the other is 0, or the label
# is 0 in all other cases. The example became famous by the fact that a single neuron, cannot learn this simple function.
# Hence, we will learn how to build a small neural network that can learn this function.

# +
#
#