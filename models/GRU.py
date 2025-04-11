from torch import nn

class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, pred_len):
    super().__init__()
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_size, pred_len)
    self.name = "GRU"

  def forward(self, x):
    out, _ = self.gru(x)
    out = self.fc(out[:, -1, :])  # Get the last time step
    return out
