# To build, use python3 -m nuitka digit_recognition.py


import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# First, we need to define the transforms on our data
# This will involve a conversion to the Pytorch Tensor format,
#   which makes MMA instructions easier. The transforms from
#   torchvision assume images (hence the name), meaning we will
#   get our tensor separated into RGB values and brightness
#   We then will normalize the tensor to a particular mean and
#   standard deviation ()
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5), (0.5))])

# Define training dataset


# Get the testing data as well (it must be different to ensure correct learning)


# Load in the training and testing data to the model with loader objects

# Now it's time to create a Neural Network Model
# First, let's define the models basic parameters, like input layer size, etc.
input_size = 50176 # This corresponds with pixel count
hidden_layer_sizes = [128, 64] # This indicates the number of neurons on the hidden layers
output_size = 7 # Races to identify: White, Hispanic, Black, East Asian, South-East Asian, Indian, Middle Eastern

# Now create the model, defining each layers relationship to the next,
#   and the activation function (ReLU in this case)


# Start the training process with the information we have
# The optimizer uses derivatives to find the fastest known learning path


# Start training, capture start time. An epoch is a pass through a dataset
    # Train with each image in the trainloader

        # Flatten the images to be managable
        

        # Pass through training
        # Set current gradients to 0 to reset prior best learning path
        

        # Start backpropagation and optimize weights
        

        # Realculate the running loss in this epoch
        
    # Print the results of this past epoch
    
print("Training complete!\n")

# Now that training is complete, run inference with the test data

# Calculate the probability of the model being correct

# Save test image for viewing

# Save the model