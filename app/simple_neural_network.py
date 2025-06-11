
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Creating dummy dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set up the model parameters
input_size = 10
hidden_size = 32
output_size = 2

# Create the model
model = SimpleNN(input_size, hidden_size, output_size)
print(f"Model architecture:\n{model}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

print("\nStarting training...")

# Training loop
for epoch in range(num_epochs):
    model.train()

    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

print("Training completed!")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)

    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = 100 * correct / total

    print(f'\nTest Accuracy: {accuracy:.2f}%')

# Save the trained model
model_path = 'simple_neural_network.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved as '{model_path}'")

complete_model_path = 'complete_model.pt'
torch.save(model, complete_model_path)
print(f"Complete model saved as '{complete_model_path}'")

import pickle
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved as '{scaler_path}'")

print("\nExample of loading the saved model:")
print("loaded_model = SimpleNN(10, 32, 2)")
print("loaded_model.load_state_dict(torch.load('simple_neural_network.pt'))")
print("loaded_model.eval()")
