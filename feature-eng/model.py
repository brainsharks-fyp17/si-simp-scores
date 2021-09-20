import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# hyper params
n_features = 2
batch_size = 7
learning_rate = 1e-5
epochs = 100
inputs = np.array(
    [[5.295051783659378, 5.876916140667268],
     [5.397040492179018, 6.815291860783776],
     [5.38165331498501, 7.718892261001518],
     [5.435080971659919, 8.800997683947978],
     [5.472035122610181, 8.724668521853003],
     [5.5120960003070785, 9.172835702653932],
     [5.6321033118823545, 9.149064095292117]],
    dtype='float32')
test_inputs = np.array([
    [5.488124313102943, 7.392462198149402],
    [5.927332223211866, 14.84080459770115]],
    dtype='float32')
# targets: Grade levels
targets = np.array([[2], [3], [4], [5], [6], [7], [9]], dtype='float32')
inputs = torch.from_numpy(inputs)
test_inputs = torch.from_numpy(test_inputs)
targets = torch.from_numpy(targets)
# Define dataset
train_ds = TensorDataset(inputs, targets)
# Define data loader
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
model = nn.Linear(in_features=n_features, out_features=1)
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Define loss function
loss_fn = F.mse_loss


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))


# Train the model for 100 epochs
fit(epochs, model, loss_fn, opt)

# Generate predictions
preds = model(test_inputs)
print(preds)

for name, param in model.named_parameters():
    print(name, param)
