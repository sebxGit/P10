import torch

class MLP(torch.nn.Module):
  def __init__(self, num_features, seq_len, pred_len, hidden_size=25):
    super().__init__()
    self.name = "MLP"

    self.all_layers = torch.nn.Sequential(
      torch.nn.Linear(num_features, seq_len),
      torch.nn.ReLU(),
      torch.nn.Linear(seq_len, hidden_size),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size, pred_len),
    )

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    logits = self.all_layers(x)
    return logits
