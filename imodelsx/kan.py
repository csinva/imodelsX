'''Code for KanLinear and KAN taken from https://github.com/Blealtan/efficient-kan/tree/master
Original implementation at https://github.com/KindXiaoming/pykan
Original paper: "KAN: Kolmogorov-Arnold Networks" https://arxiv.org/abs/2404.19756
'''

import torch
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1,
                               self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(
            splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] +
                        2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + \
            (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1,
                               device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1,
                               device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KANsklearn(BaseEstimator):
    def __init__(self, hidden_layer_size=64,
                 device='cpu', regularize_activation=1.0, regularize_entropy=1.0):
        '''
        Params
        ------
        hidden_layer_size : int
            Number of neurons in the hidden layer.
        '''
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.regularize_activation = regularize_activation
        self.regularize_entropy = regularize_entropy

    def fit(self, X, y, batch_size=512):
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
            num_outputs = len(self.classes_)
            y = torch.tensor(y, dtype=torch.long)
        else:
            num_outputs = 1
            y = torch.tensor(y, dtype=torch.float32)
        X = torch.tensor(X, dtype=torch.float32)
        num_features = X.shape[1]
        self.model = KAN(
            layers_hidden=[num_features,
                           self.hidden_layer_size, num_outputs]
        ).to(self.device)

        X_train, X_tune, y_train, y_tune = train_test_split(
            X, y, test_size=0.2, random_state=42)

        dset_train = torch.utils.data.TensorDataset(X_train, y_train)
        dset_tune = torch.utils.data.TensorDataset(X_tune, y_tune)
        loader_train = DataLoader(
            dset_train, batch_size=batch_size, shuffle=True)
        loader_tune = DataLoader(
            dset_tune, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        # Define loss
        if isinstance(self, ClassifierMixin):
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        tune_losses = []
        for epoch in tqdm(range(100)):
            self.model.train()
            for x, labs in loader_train:
                x = x.view(-1, num_features).to(self.device)
                optimizer.zero_grad()
                output = self.model(x).squeeze()
                loss = criterion(output, labs.to(self.device).squeeze())
                loss += self.model.regularization_loss(
                    self.regularize_activation, self.regularize_entropy)

                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            tune_loss = 0
            with torch.no_grad():
                for x, labs in loader_tune:
                    x = x.view(-1, num_features).to(self.device)
                    output = self.model(x).squeeze()
                    tune_loss += criterion(output,
                                           labs.to(self.device).squeeze()).item()
            tune_loss /= len(loader_tune)
            tune_losses.append(tune_loss)

            # Update learning rate
            scheduler.step()

            # print(
            #     f"Epoch {epoch + 1}, Tune Loss: {val_loss:.4f}"
            # )

            # apply early stopping
            if len(tune_losses) > 3 and tune_losses[-1] > tune_losses[-2]:
                print("Early stopping")
                return self

        return self

    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        if isinstance(self, ClassifierMixin):
            return self.classes_[output.argmax(dim=1).cpu().numpy()]
        else:
            return output.cpu().numpy()


class KANClassifier(KANsklearn, ClassifierMixin):
    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        return torch.nn.functional.softmax(output, dim=1).cpu().numpy()


class KANRegressor(KANsklearn, RegressorMixin):
    pass


if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2)
    model = KANClassifier(hidden_layer_size=64, device='cpu',
                          regularize_activation=1.0, regularize_entropy=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)
    print('Test acc', accuracy_score(y, y_pred))

    # now try regression
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=2)
    model = KANRegressor(hidden_layer_size=64, device='cpu',
                         regularize_activation=1.0, regularize_entropy=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)
    print('Test correlation', np.corrcoef(y, y_pred.flatten())[0, 1])
