import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(num_inputs, 8)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.hidden2 = nn.Linear(8, 32)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.output = nn.Linear(32, 1)
        self.act_output = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout1(self.act1(self.hidden1(x)))
        x = self.dropout2(self.act2(self.hidden2(x)))
        x = self.act_output(self.output(x))
        return x
    
    def train_model(self, train_loader, loss_fn, optimizer, n_epochs, device):
        losses = []
        accuracies = []
        for epoch in range(n_epochs):
            self.train()
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                predictions = (y_pred > 0.5).float()
                correct_predictions += (predictions == y_batch).sum().item()
                total_predictions += y_batch.size(0)

            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            print(f'Completed Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.4f}')
        return losses, accuracies
    
def set_seed(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_nn_progress(losses, accuracies):
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy', color='orange')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()