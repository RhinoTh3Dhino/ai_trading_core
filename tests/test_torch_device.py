import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Fundet device:", device)
if torch.cuda.is_available():
    print("CUDA device navn:", torch.cuda.get_device_name(0))
    print("RAM brugt (MB):", torch.cuda.memory_allocated() // (1024**2))
else:
    print("Ingen CUDA-GPU fundet – kører på CPU")
