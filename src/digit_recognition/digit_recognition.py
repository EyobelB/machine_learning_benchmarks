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

# Download the MNIST training dataset remotely
training_data = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)

# Get the testing data as well (it must be different to ensure correct learning)
testing_data = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

# Load in the training and testing data to the model with loader objects
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=64, shuffle=True)

# Now it's time to create a Neural Network Model
# First, let's define the models basic parameters, like input layer size, etc.
input_size = 784 # This corresponds with pixel count
hidden_layer_sizes = [128, 64] # This indicates the number of neurons on the hidden layers
output_size = 10 # Only 10 digits to identify, so 10 possible outcomes

# Now create the model, defining each layers relationship to the next,
#   and the activation function (ReLU in this case)
model = nn.Sequential(
    nn.Linear(input_size, hidden_layer_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_layer_sizes[1], output_size),
    nn.LogSoftmax(dim=1) # Used to define output bounds
)

# Start the training process with the information we have
# The optimizer uses derivatives to find the fastest known learning path
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
loss_function = nn.NLLLoss()


# Start training, capture start time. An epoch is a pass through a dataset
start_time = time()
for epoch in range(0, 15):
    running_loss = 0
    # Train with each image in the trainloader
    for images, labels in training_loader:
        # Flatten the images to be managable
        images = images.view(images.shape[0], -1)

        # Pass through training
        optimizer.zero_grad() # Set current gradients to 0 to reset prior best learning path
        output = model(images)
        loss = loss_function(output, labels)

        # Start backpropagation and optimize weights
        loss.backward()
        optimizer.step()

        # Realculate the running loss in this epoch
        running_loss += loss.item()
    # Print the results of this past epoch
    print(f"Epoch {epoch} - Training loss: {running_loss/len(training_loader)}")
    print(f"Time since start in minutes: {(time()-start_time)/60}\n")
print("Training complete!\n")

# Now that training is complete, run inference with the test data
images, labels = next(iter(testing_loader))
test_image = images[0].view(1, 784)
with torch.no_grad():
    probability_logistic = model(test_image)

# Calculate the probability of the model being correct
probability_correct = torch.exp(probability_logistic)
probability_correct = list(probability_correct.numpy()[0])
print("Predicted Digit: ", probability_correct.index(max(probability_correct)))

# Save test image for viewing
save_image(images[0], "test.png")

# Save the model
torch.save(model, 'digit_recognition.pt')