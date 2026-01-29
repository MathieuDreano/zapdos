import sklearn.model_selection
import torch
import requests
from torch import nn
import pandas as pd
import numpy as np
import torchvision
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("already exists")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

print(torch.__version__)
print(f"cuda available: {torch.cuda.is_available()}")


'''
    My model
'''
class Exodus(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)


'''
    Compute accuracy based on prediction and ground truth vectors
'''
def acc_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

'''
    Train the model over several epochs.
'''
def train(model, X_train, X_test, y_train, y_test, loss_fn, optimizer, epochs):
    epoch = 0
    done = False

    while not done:
        # pass input to model
        y_logits = model(X_train).squeeze()
        # normalize logits to get predictions from 0 to 1
        y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> pred probs -> pred labels
        # calculate the loss (the less the better)
        loss = loss_fn(y_logits, y_train)
        # calculate accuracy
        acc = acc_fn(y_true=y_train, y_pred=y_pred)
        # resets the gradients of all model parameters to zero.
        optimizer.zero_grad()
        # computes gradients of the loss function with respect to the model parameters using backpropagation
        loss.backward()
        # updates the model parameters based on the computed gradients and the chosen optimization algorithm
        optimizer.step()


        # check how good the model is against the testing data, should improve over time
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = acc_fn(y_true=y_test, y_pred=test_pred)

        # Print current result
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {test_loss:.5f}, Accuracy: {test_acc:5f}')

        # display graph to a visual representation of how good the model is doing
        if epoch % 100 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Train")
            plot_decision_boundary(model, X_train, y_train)
            plt.subplot(1, 2, 2)
            plt.title("Test")
            plot_decision_boundary(model, X_test, y_test)
            plt.show()

        epoch += 1
        # early stop or epochs reached
        if test_acc > 99.9 or epoch == epochs:
            done = True

if __name__ == '__main__':

    # create training and testing datasets
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = (sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42))

    # instantiate a model
    model = Exodus()
    # using binary cross entropy (generally used for classification)
    loss_fn = nn.BCEWithLogitsLoss()
    # standard gradiant descent (SGD)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    # train model over 1000 epochs
    epochs = 1000
    train(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, loss_fn=loss_fn,
          optimizer=optimizer, epochs=epochs)
