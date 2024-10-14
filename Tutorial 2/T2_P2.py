###########
# IMPORTS #
###########
import torch

torch.manual_seed(42)

#########################################################################################
# ------------- SECTION 2: DYNAMIC COMPUTATION GRAPH AND BACKPROPAGATION -------------  #
#########################################################################################
# PyTorch can automatically calculate the gradients/derivatives of functions that we define. 
# 
# Neural networks consist of layers of neurons, each with associated weights and biases.
# - **Weights**: These are parameters that define the strength of the connection between 
#   neurons. They are multiplied by the input data and determine the output of each neuron.
# - **Biases**: These are added to the weighted sum of inputs to allow the network to fit
#   the data more flexibly. They help shift the activation function, allowing the model to
#   better fit the given data.
#
# During training, the network adjusts these weights and biases to minimize the error in its
# predictions. This process is called "learning."
#
# Why do we want gradients (Backpropagation):
# Consider we have defined a function, a neural net, that is supposed to compute a certain a
# output y for an input vector x. We then define an error that tells us how wrong our network
# predicted the actual value of y from the input x. Based on this error we can use the gradients
# to update the weights W that were responsible for the output, so that the next time we present
# input x to our network, the output will be closer to what we want.

# We can specify which tensors require gradients. By default, tensors don't require gradients.
# We can check if a initiated tensor requires gradients via the function .requires_grad called on the tensor
x = torch.tensor([1,2,3,4])
print(x.requires_grad)

# The function requires_grad_() is an inplace operation which can be used to 'make' tensors require gradients.
# The input argument 'required_grad=True' can be passed to most tensor functions during initalisation

###########################
# 2.1 COMPUTATIONAL GRAPH #
###########################
# Creating a computational graph for the function:
# y = ( 1/l(x) ) * Σ [ (xi + 2)^2 + 3 ]
# l(x) used to denote the number of elements in x. We are taking the mean here over the operation within the sum
# Imagine that x are our parameters, and we want to optimise (either maximise of minimise) the output y. For this, 
# we want to obtain the gradients dy/dx. For the example, we'll use x = [0,1,2] as input.

x = torch.arange(3, dtype=torch.float32, requires_grad=True) # ONLY FLOAT TENSORS CAN HAVE GRADIENTS, Therefore we use dtype to cast to float32.
print("X", x)

# +----------------------------------+
# | BUILDING THE COMPUTATIONAL GRAPH |
# +----------------------------------+
a=x+2
b=a**2
c=b+3
y=c.mean()
print("Y",y)

# "a" is calculated based on the inputs "x" and the constant 2
# "b" is "a" squared
# "c" is "b" plus 3
# "y" is the mean of c

# Each node of the computational graph has automatically defined a function for calculating the gradients with respect to its inputs - grad_fn
# We can perform backpropagation on the computation graph by calling the function backward() on the last output, which effectively calculates the
# gradients of each tensor that has the argument requires_grad=True. 
y.backward()
# After performing backpropagation using the function .backward(), the x.grad now contains the gradient dy/dx, the gradient indicates how a change
# in x will affect the output y given the current input x.
print(x.grad)
