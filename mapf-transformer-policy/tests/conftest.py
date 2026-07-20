import torch

# Avoid pathological thread oversubscription in small CPU smoke tests.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
