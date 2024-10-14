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
# become cumbersome quickly,