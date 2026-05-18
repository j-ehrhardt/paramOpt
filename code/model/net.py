import torch
import torch.nn as nn
import pickle

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.ln1(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out + identity)
        return out


class ResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1):
        super(ResidualNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc_in(x)
        out = self.res_blocks(out)
        out = self.fc_out(out)
        return out


class HeteroscedaticResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1):
        super(HeteroscedaticResidualNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.mu_out = nn.Linear(hidden_dim, 1)
        self.logvar_out = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.logvar_out.bias, -2.0)

    def forward(self, x):
        h = self.fc_in(x)
        h = self.res_blocks(h)
        mu = self.mu_out(h)
        logvar = self.logvar_out(h)
        return torch.cat([mu, logvar], dim=1)


class MixtureDensityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1):
        super(MixtureDensityNet, self).__init__()
        K = 8
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.mu_out = nn.Linear(hidden_dim, K)
        self.logvar_out = nn.Linear(hidden_dim, K)
        nn.init.constant_(self.logvar_out.bias, -2.0)
        self.pi_out = nn.Linear(hidden_dim, K)

    def forward(self, x):
        h = self.fc_in(x)
        h = self.res_blocks(h)
        mu = self.mu_out(h)
        logvar = self.logvar_out(h)
        pi = torch.softmax(self.pi_out(h), dim=1)
        return torch.cat([mu, logvar, pi], dim=1)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(Net, self).__init__()
        act = nn.ReLU()
        self.net = nn.Sequential()
        for layer in range(n_layers - 1):
            if layer == 0:
                self.net.add_module(f'layer_{layer}', nn.Linear(input_dim, hidden_dim))
            else:
                self.net.add_module(f'layer_{layer}', nn.Linear(hidden_dim, hidden_dim))
            self.net.add_module(f'activation_{layer}', act)
            self.net.add_module(f'dropout_{layer}', nn.Dropout(dropout))
        self.net.add_module(f'layer_{n_layers}', nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def save_model(model, path, model_type='ff'):
    if 'tabpfn' in model_type:
        with open(f'{path}/model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        torch.save(model, f'{path}/model.pkl')
    print(f'Model saved under {path}/model.pkl')


def load_model(path, model_type='ff'):
    try:
        with open(f'{path}/model.pkl', 'rb') as f:
            if 'tabpfn' in model_type:
                model = pickle.load(f)
            else:
                model = torch.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {path}: {str(e)}")