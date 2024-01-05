from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from tqdm.auto import trange

from .data_processing import convert_df_to_utility_mat, normalize_utility_mat


class MatrixFactorization(BaseEstimator):
    def __init__(
        self,
        k: int = 2,
        user_based: bool = True,
        lr: float = 0.5,
        lam: float = 0.1,
        max_iter: int = 1000,
        print_every: int = 100,
        epsilon: float = 1e-6,
        random_state: int = 42,
        verbose: int = 1,
        **kwargs,
    ):
        self.k = k
        self.user_based = user_based
        self.raw_data = None

        # Params for MF
        self.X = None
        self.W = None
        self.n_users = -1
        self.n_items = -1
        self.n_ratings = -1
        self.utility_matrix = None
        self.mat_mask = None
        self.rating_means = None

        # Learning hyperparams
        self.lr = lr
        self.lam = lam  # For regularization
        self.max_iter = max_iter
        self.print_every = print_every
        self.epsilon = epsilon

        # Others
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)
        self.verbose = verbose

    def init_params(
        self, Xinit: Optional[np.array] = None, Winit: Optional[np.array] = None
    ):
        self.X = Xinit
        self.W = Winit

    def loss(self) -> float:
        delta = self.utility_matrix - self.X.dot(self.W)
        delta = np.ma.array(delta, mask=self.mat_mask)
        L = 0.5 * (delta**2).mean()
        L += (
            0.5
            * self.lam
            * (np.linalg.norm(self.X, "fro") + np.linalg.norm(self.W, "fro"))
        )
        return L

    def update_X(self) -> None:
        grad_X = (
            -((self.utility_matrix - self.X.dot(self.W)).dot(self.W.T)) / self.n_ratings
            + self.lam * self.X
        )
        self.X -= self.lr * grad_X

    def update_W(self) -> None:
        grad_W = (
            -np.mean(self.X.T.dot(self.utility_matrix - self.X.dot(self.W)))
            / self.n_ratings
            + self.lam * self.W
        )
        self.W -= self.lr * grad_W

    def fit(self, X, y=None, **kwargs):
        self.raw_data = X
        utility_mat = convert_df_to_utility_mat(X)
        self.utility_matrix, self.rating_means = normalize_utility_mat(
            utility_mat, user_based=self.user_based
        )
        self.mat_mask = utility_mat == 0.0

        self.n_items, self.n_users = self.utility_matrix.shape
        self.n_ratings = np.sum(~self.mat_mask)

        if self.X is None or self.X.shape != (self.n_items, self.k):
            self.X = self.rng.random((self.n_items, self.k))

        if self.W is None or self.W.shape != (self.k, self.n_users):
            self.W = self.rng.random((self.k, self.n_users))

        pbar = trange(self.max_iter, desc="Iteration ", disable=not self.verbose)
        prev_loss = 1e9
        for i in pbar:
            self.update_X()
            self.update_W()
            loss = self.loss()
            pbar.set_description(f"Iteration {i} Loss {loss}")
            loss_diff = np.abs(loss - prev_loss)
            if loss_diff < self.epsilon:
                # print(
                #     f"Training loss improved by {loss_diff}. Converged at iteration {i}."
                # )
                break
            prev_loss = loss
            # if (i + 1) % self.print_every == 0:
            #     print(f"Iteration {i + 1} Loss {loss} RMSE {self.evaluate_RMSE()}")
            # time.sleep(0.5)
        # print("Traning completed!")

    def predict_single(
        self, u: int, i: Optional[int] = None, bound: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        predict the rating of user u for item i
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.rating_means[u]
        else:
            bias = self.rating_means[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias

        # Truncation if bound is given
        if bound is not None:
            pred = np.clip(pred, bound[0], bound[1])
        pred = int(pred)
        return pred

    def predict(self, X, bound: Optional[Tuple[int, int]] = None) -> float:
        y_pred = []
        for _, row in X.iterrows():
            pred = self.predict_single(row["user"], row["item"], bound=bound)
            y_pred.append(pred)
        y_pred = np.array(y_pred)
        return y_pred

    def score(self, X, *args, **kwargs) -> float:
        y_pred = self.predict(X)
        y_true = X["target"].values.astype(int)
        return -1 * mean_squared_error(y_true, y_pred, squared=False)
