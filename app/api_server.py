from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import List

app = FastAPI()

# Neural network architecture
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

# Input model
class PredictionInput(BaseModel):
    features: List[float]

# Load model and scaler
model = SimpleNN(10, 32, 2)
model.load_state_dict(torch.load('app/simple_neural_network.pt'))
model.eval()

with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Process input
    features = np.array(input_data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)

    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": predicted.item()}

# vvv Use only while testing locally vvv

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
