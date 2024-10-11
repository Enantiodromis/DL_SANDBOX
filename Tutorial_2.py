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

# Printing torch version
print("Using torch", torch.__version__)
torch.manual_seed(42)

###################################################
# ------------- SECTION 1: TENSORS -------------  #
###################################################

##############################
# 1.1 TENSOR INITIALISATION  #
##############################
# Tensors are equivalent to Numpy arrays, which also have GPU acceleration support. 
# - A vector is a 1-D tensor
# - A matrix is a 2-D tensor

# There are many ways to create a Tensor, the simplest is to call the Tensor function via "torch.Tensor"
x = torch.Tensor(2, 3, 4)
print(f"\nExample of 'torch.Tensor':\n{x}")

# "torch.Tensor" allocates memory for the desired tensor, but reuses any values that have already been in memory. 
# To directly assign values to the tensor during initialisation, there are many alternatives:
# "torch.zeros" --> Creates a tensor filled with 
print(f"\nExample of 'torch.zeros':\n{torch.zeros(10)}")

# "torch.ones" --> Creates a tensor filled with ones
print(f"\nExample of 'torch.ones':\n{torch.ones(10)}")

# "torch.rand" --> Creates a tensor with random values uniformaly sampled between 0 and 1
print(f"\nExample of 'torch.rand':\n{torch.rand(10)}")

# "torch.randn" --> Creates a tensor with random values sampled from a nomrmal/gaussian distribution with mean 0 and variance 1
print(f"\nExample of 'torch.randn':\n{torch.randn(10)}")

# "torch.arange" --> Creates a tensor containing the values N, N+1, N+2,..., M
print(f"\nExample of 'torch.arange':\n{torch.arange(10)}")

# You can also pass the Tensor function a list, which will be convereted to a Tensor representation
example_list = [[1,2], [3,4], [5,6]]
list_to_tensor = torch.Tensor(example_list)
print(f"\nA tensor, created by passing a list as input to the 'torch.Tensor' function:\n{list_to_tensor}")

############################
# 1.2 TENSOR SHAPE & SIZE  #
############################
# The '.shape' or '.size' functions can be called to determine the shape of a tensor.

# Example of a 1 dimensional tensor:
one_dim_list = [[1,2,3,4]] # Defining a 1 dimensional list
one_dim_tensor = torch.Tensor(one_dim_list) # Converting list to tensor
tensor_shape_using_shape = one_dim_tensor.shape
tensor_shape_using_size = one_dim_tensor.size()

print(f"\nThe shape of the ONE dimensional tensor {one_dim_tensor}, using the function '.shape': {tensor_shape_using_shape} and '.size': {tensor_shape_using_size}")

# Example of a 2 dimensional tensor:
two_dim_list = [[1,2],[3,4]] # Defining a two dimensional list
two_dim_tensor = torch.Tensor(two_dim_list) # Converting list to tensor
tensor_shape_using_shape = two_dim_tensor.shape
tensor_shape_using_size = two_dim_tensor.size()

print(f"\nThe shape of the TWO dimensional tensor {two_dim_tensor}, using the function '.shape': {tensor_shape_using_shape} and '.size': {tensor_shape_using_size}")

#################################################
# 1.3 CONVERTING BETWEEN TENSOR AND Numpy ARRAY #
#################################################
# Tensors can be converted to numpy arrays and numpy arrays back to tensors

# To convert FROM Numpy TO tensor use the function '.from_numpy' in the form 'torch.from_numpy(numpy array)'
np_array = np.array([[1,2],[3,4]])
np_to_tensor = torch.from_numpy(np_array)
print("\nCONVERSION FROM NUMPY TO TENSOR")
print(f"\nVariable prior to conversion {np_array}")
print(f"Variable type prior to conversion: {type(np_array)}")
print(f"\nVariable after conversion {np_to_tensor}")
print(f"Variable type after conversion: {type(np_to_tensor)}")

# To convert FROM tensor TO Numpy use the function '.numpy()' in the form 'tensor.numpy()'
tensor_for_conversion = torch.Tensor([[1,2],[3,4]])
converted_tensor = tensor_for_conversion.numpy()
print("\nCONVERSION FROM TENSOR TO NUMPY")
print(f"\nVariable prior to conversion {tensor_for_conversion}")
print(f"Variable type prior to conversion: {type(tensor_for_conversion)}")
print(f"\nVariable after conversion {converted_tensor}")
print(f"Variable type after conversion: {type(converted_tensor)}")

#######################################
# 1.4 (FUN)DEMENTAL TENSOR OPERATIONS #
#######################################

# OPERATION 1 --> ADDITION
x1 = torch.ones(3)
x2 = torch.tensor([1,2,3])
y = x1 + x2 # Performing the tensor addition
print(f"\nTensor x1: {x1} added to x2: {x2} = {y}")

# Side note: addition can also be performed in-place via 'x1.add_(x2)'

# OPERATION 2 --> VIEW
# The shape of tensors can be re-organised (as long as the new shape has the same number of elements) using the '.view' function.
# Eg: A 1-D tensor x1 = [1, 2, 3, 4] can be re-organised into a 2-D tensor of size 2 such as x2 = [[1,2],[3,4]]

x1 = torch.arange(1,5,1)
x2 = x1.view(2,2)
x3 = x1.view(4,1)
print(f"\nThe 1-D tensor {x1} can be re-organised into a 2-D vector of size 2 {x2}, a 4-D vector of size 1 {x3} etc...")

# OPERATION 3 --> PERMUTE
# The permute function reorders the dimensions of a tensor according to the specified order of indics using the '.permute' function.
# Eg: A 2-D tensor x1 = [[1, 2, 3], [4, 5, 6]] can be re-organised by swapping its dimensions using permute to produce x2 = [[1, 4], [2, 5], [3, 6]]

x1 = torch.Tensor([[1,2,3],[4,5,6]])
x2 = x1.permute(1, 0)
print(f"\nThe 2-D tensor {x1} can be re-organized by swapping its dimensions into {x2}")

# OPERATION 4 --> MULTIPLICATION
# The torch.matmul() function is the most commonly used matrix multiplication function in PyTorch due to its versatility.
# It can handle different tensor dimensions (2D, vector-matrix, batched matrix multiplication) and supports broadcasting.
# Eg: A 2-D tensor x1 = [[1, 2], [3, 4]] and x2 = [[5, 6], [7, 8]] can be multiplied using torch.matmul(x1, x2) to produce [[19, 22], [43, 50]].

x1 = torch.Tensor([[1,2],[3,4]])
x2 = torch.Tensor([[5,6],[7,8]])
multiplied_tensors = torch.matmul(x1, x2)
print(f"\nThe tensors {x1} and {x2} multipled via matmul = {multiplied_tensors}")

########################
# 1.5 INDEXING TENSORS #
########################
# Indexing allows for the selection of specific parts of a tensor.
x = torch.arange(1,21,1).view(4,5)
# Given tensor [[1,2,3,4,5],[6,7,8,9,10]]

# A column can be chosen via [:,n] where n is the column number (zero indexed)
print(f"\nFor tensor {x} the second column is {x[:,1]}")

# A row can be chosen via [n] where n is the row number (zero indexing)
print(f"\nFor tensor {x} the first row is {x[0]}")

# Multiple rows can be chosen via [:n] where n is the index of the row you want to select until.
print(f"\nFor tensor {x} the first 3 rows is {x[:3]}")

# Multiple rows of a column can be chosen via [:n, -1] where n is the row you want to select until and -1 will get the last column.
print(f"\nFor tensor {x} the first 3 rows of the last column is {x[:3, -1]}")

# Specific row indexes can be chosen via [1:3, :]
print(f"\nFor tensor {x} the second to third are {x[1:3, :]}")