import torch
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from imodelsx.kan.kan_modules import KANModule, KANGAMModule
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score


class KAN(BaseEstimator):
    def __init__(self, hidden_layer_size=64, device='cpu',
                 regularize_activation=1.0, regularize_entropy=1.0, regularize_ridge=0.0):
        '''
        Params
        ------
        hidden_layer_size : int
            Number of neurons in the hidden layer.
        regularize_ridge: float
            Ridge regularization parameter (only applies to KANGAM)
        '''
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.regularize_activation = regularize_activation
        self.regularize_entropy = regularize_entropy
        self.regularize_ridge = regularize_ridge

    def fit(self, X, y, batch_size=512, lr=1e-3, weight_decay=1e-4, gamma=0.8):
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
        if isinstance(self, (KANGAMClassifier, KANGAMRegressor)):
            self.model = KANGAMModule(
                num_features=num_features,
                hidden_layer_size=self.hidden_layer_size,
                n_classes=num_outputs,
            ).to(self.device)
        else:
            self.model = KANModule(
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
                                lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

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
                if isinstance(self, (KANGAMClassifier, KANGAMRegressor)):
                    loss += self.model.regularization_loss(
                        self.regularize_activation, self.regularize_entropy, self.regularize_ridge)
                else:
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
            scheduler.step()

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


class KANClassifier(KAN, ClassifierMixin):
    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        return torch.nn.functional.softmax(output, dim=1).cpu().numpy()


class KANRegressor(KAN, RegressorMixin):
    pass


class KANGAMClassifier(KANClassifier):
    pass


class KANGAMRegressor(KANRegressor):
    pass


if __name__ == '__main__':
    # classification
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=2)
    for m in [KANClassifier, KANGAMClassifier]:
        model = m(hidden_layer_size=64, device='cpu',
                  regularize_activation=1.0, regularize_entropy=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        print('Test acc', accuracy_score(y, y_pred), m)

    # regression
    X, y = make_regression(n_samples=1000, n_features=8, n_informative=2)
    for m in [KANRegressor, KANGAMRegressor]:
        model = m(hidden_layer_size=64, device='cpu',
                  regularize_activation=1.0, regularize_entropy=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        print('Test correlation', np.corrcoef(y, y_pred.flatten())[0, 1], m)