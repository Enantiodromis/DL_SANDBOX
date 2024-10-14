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

# +----------------------+
# | 3.1 MODEL DEFINITION |
# +----------------------+
import torch.nn as nn # The toch.nn package is used to define, network layers, activation layers, loss functions etc...
import torch.nn.functional as F

# In PyTorch a neural network is built up out of modules. Modules contain other modules, a neural network is considered to be a module itself.
class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Function for performing the calculation of the module.
        pass
# +------------------------------------+
# | 3.1.1 DEFINING A SIMPLE CLASSIFIER |
# +------------------------------------+
# We now define a small neural network, with an input layer, one hidden layer with tanh as a activation function, and a output layer. 

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialising the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perfrom the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

# As we are performing binary classification, we will use a single output neuron. 
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
print(model) # Printing a module shows all its submodules.
for name, param in model.named_parameters(): # The parameters of a module can be obtained using its parameter()/named_parameters()
    print(f"Parameter {name}, shape {param.shape}")

# Each linear layer has a weight matrix of the shape [output, input], and a bias of the shape [output].
# The tanh activation function does not have any parameters.

# -----------+
# | 3.2 DATA |
# +----------+
import torch.utils.data as data # PyTorch provides functionalities to load the training and test data efficiently
# data.Dataset: The dataset class provides a uniform interface to access the training/test data.
# data.DataLoader: The data loader makes sure to efficiently load and stack the data points from the dataset into batches during training

# +--------------------------+
# | 3.2.1 DEFINING A DATASET |
# +--------------------------+
# To define a dataset in PyTorch, we simply specify two functions: __getitem__, and __len__. The get-item function has to return to the i-th
# data point in the dataset, while len function returns the size of the dataset. For the XOR dataset, we can defien the dataset class as follows:

class XORDataset(data.Dataset):

    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

dataset = XORDataset(size=200)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])

def visualize_samples(data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

visualize_samples(dataset.data, dataset.label)
plt.show()

# +------------------------+
# | 3.2.2 LOADING THE DATA |
# +------------------------+
# The DataLoader class represents a Python iterable over a dataset with support for automatic batching amongst other features.
# The DataLoader communicates with the dataset using the function __getitem__, and stacks its outputs as tensors over the first
# dimension to form a batch. We usually don't have to define our own data loader class, but can just create an object of it with
# the dataset as input. 
#
# Some main parameters:
# batch_size --> Number of samples to stack per batch
# shuffle --> If TRUE, the data is returned in a random order. This is important during training for introducing stochasticity.
# num_workers --> Number of subprocesses to use for data loading. The default 0, means that data will be loaded in the main process
#                 which can slow down training for datasets where loading a data point takes considerable amount of time.
# pin_memory --> If TRUE, the data loader will copy Tensors into CUDA pinned memory before returning them. Usually good practice for training. 
# drop_last --> If TRUE, the last batch is dropped in case it is smaller than the specified batch size. This occurs when the dataset size is not
#               a multiple of the batch size. 

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)

# next(iter(...)) catches the first batch of the data loader
# If shuffle is True, this will return a different batch every time we run this cell
# For iterating over the whole dataset, we can simple use "for batch in data_loader:..."
data_inputs, data_labels = next(iter(data_loader))

# The shape of the outputs are [batch_size, d_1,...,d_N] where d_1,...,d_N are the
# dimensions of the data point returned from the dataset class
print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)

# +------------------+
# | 3.3 OPTIMISATION |
# +------------------+
# During training, we will perfor the following steps:
# 1. Get a batch from the data loader
# 2. Obtain the predictions from the model for the batch
# 3. Calculate the loss based on the difference between predictions and labels
# 4. Backpropagation: calculate the gradients for every parameter with respect to the loss
# 4. Update the parameters of the model in the direction of the gradients

# +--------------------+
# | 3.3.1 LOSS MODULES |
# +--------------------+
# PyTorch already provides a list of predefined loss functions which we can use. For Binary cross entropy, PyTorch has two modules:
# nn.BCELoss() and nn.BCEWithLogitsLoss(). The nn.BCEWithLogitsLoss() function is more stable than using plain Sigmoid followed by
# a BCE loss because of the logarithms applied in the loss function. It is adviced to use loss functions applied on "logits" where possible.
loss_module = nn.BCEWithLogitsLoss()

# +-----------------------------------+
# | 3.3.2 STOCHASTIC GRADIENT DESCENT |
# +-----------------------------------+
# For updating the parameters, PyTorch provides the package torch.optim that has most popular optimisers implemented.
# For now, Stochastic Gradient Descent (SGD) will be used. SGD updates parameters by multipyling the gradients with a
# small constant, called learning rate, and subtracting those from the parameters (hence minimising the loss). We therefore
# slowly move towards the direction of minimising the loss. A good default value of the learning rate for a small network as ours is 0.1

# Input to the optimiser are the parameters of the model: model.parameters()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

# The optimiser provides two useful functions:
# optimiser.step() --> The step function updates the parameters based on the gradients as explained above.
# optimiser.zero_grad() --> Sets the gradients of all parameters to zero.(A crucial pre-step before performing backpropagation).
#                           If we call the backward function on the loss while the parameter gradients are non-zero from the
#                           previous batch, the new gradients would actually be added to the previous ones instead of overwriting them.
#                           This is done because a parameter might occur multiple times in a computation graph, and we need to sum the
#                           gradients in this case instead of replacing them.

# +--------------+
# | 3.4 TRAINING |
# +--------------+
# We now create a slightly larger dataset and specify a data loader with a larger batch size.
train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

train_model(model, optimiser, train_data_loader, loss_module)

# +------------------------------+
# | 3.5 MODEL SAVING AND LOADING |
# +------------------------------+
# In order to save the model, we extract the so-called state_dict from the model which contains all learnable parameters
state_dict = model.state_dict()
print(state_dict)

# torch.save(object, filename). For the filename, any extension can be used
torch.save(state_dict, "our_model.tar")

# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("our_model.tar")

# Create a new model and load the state
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)

# Verify that the parameters are the same
print("Original model\n", model.state_dict())
print("\nLoaded model\n", new_model.state_dict())

# +----------------------+
# | 3.6 MODEL EVALUATION |
# +----------------------+
# In order to evaluate the model, we create a testing dataset and a corresponding data loader
test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarise predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

@torch.no_grad() # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
def visualize_classification(model, data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(dpi=500)
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples",fontsize=8)
    plt.ylabel(r"$x_2$",fontsize=7)
    plt.xlabel(r"$x_1$",fontsize=7)
    plt.legend(fontsize=6)

    # Let's make use of a lot of operations we have learned above
    model
    c0 = torch.Tensor(to_rgba("C0"))
    c1 = torch.Tensor(to_rgba("C1"))
    x1 = torch.arange(-0.5, 1.5, step=0.01)
    x2 = torch.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]  # Specifying "None" in a dimension creates a new one
    output_image = output_image.cpu().numpy()  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig

_ = visualize_classification(model, dataset.data, dataset.label)
plt.show()