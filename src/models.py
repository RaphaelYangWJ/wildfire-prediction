import torch.nn as nn



# === Class: Neural Networks
class network(nn.Module):
    # === Init function
    def __init__(self,
                 activation: str = "Tanh",
                 dropout_value: float = 0.1,
                 train_x: str = None
                 ):

        super().__init__()
        self.activation = activation
        self.dropout_value = dropout_value
        self.train_x = train_x

        self.layer1 = nn.Linear(self.train_x.shape[1], self.train_x.shape[1]*2)
        self.layer2 = nn.Linear(self.train_x.shape[1]*2, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 4)
        self.layer5 = nn.Linear(4, 1)
        if self.activation == "Tanh":
            self.activation = nn.Tanh()
        elif self.activation == "ReLU":
            self.activation = nn.ReLU()
        elif self.activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif self.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(self.dropout_value)

    # === Forward function
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.dropout(x)
        out = self.layer5(x)

        return out