# models/pytorch_model.py

import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=1, activation="sigmoid"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Sigmoid() if activation == "sigmoid" else nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def get_device():
    # Vælg GPU hvis muligt, ellers CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    device = get_device()
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    # Simple test – brug kun ved standalone test
    input_dim = 10
    model = MLPClassifier(input_dim)
    x = torch.randn(5, input_dim)
    print("Output shape:", model(x).shape)
    print("OK! PyTorch model virker.")
