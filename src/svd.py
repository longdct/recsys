from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from scipy.linalg import svd
from tqdm.auto import trange

from .data_processing import normalize_utility_mat


class SVD(BaseEstimator):
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
        **kwargs,
    ):
        self.K = k
        self.user_based = user_based
        self.raw_data = None

        # Params for MF
        self.X = Xinit
        self.W = Winit
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
        self.rng = np.random.default_rng(seed=random_state)

    def loss(self) -> float:
        delta = self.utility_matrix - self.X.dot(self.W)
        delta = np.ma.array(delta, mask=self.mat_mask)
        L = 0.5 * (delta**2).mean()
        L += self.lam * (np.linalg.norm(self.X, "fro") + np.linalg.norm(self.W, "fro"))
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

    def fit(self, X, y=None):
        self.raw_data = X
        self.utility_matrix, self.rating_means = normalize_utility_mat(
            X, user_based=self.user_based
        )
        self.mat_mask = X == 0.0

        self.n_items, self.n_users = self.utility_matrix.shape
        self.n_ratings = np.sum(~self.mat_mask)

        if self.X is None or self.X.shape != (self.n_items, self.K):
            self.X = self.rng.random((self.n_items, self.K))

        if self.W is None or self.W.shape != (self.K, self.n_users):
            self.W = self.rng.random((self.K, self.n_users))

        pbar = trange(self.max_iter, desc="Iteration ")
        prev_loss = 1e9
        for i in pbar:
            self.update_X()
            self.update_W()
            loss = self.loss()
            if np.abs(loss - prev_loss) < self.epsilon:
                break
            prev_loss = loss
            pbar.set_description(f"Iteration {i} Loss {loss}")
            if (i + 1) % self.print_every == 0:
                print(f"Iteration {i + 1} Loss {loss} RMSE {self.evaluate_RMSE()}")
            # time.sleep(0.5)

    def predict(
        self, u: int, i: Optional[int] = None, bound: Optional[Tuple[int]] = None
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
        return pred

    def predict_for_user(self, user_id, bound: Optional[Tuple[int]] = None) -> np.array:
        """
        predict ratings one user give all unrated items
        """
        y_pred = self.X.dot(self.W[:, user_id]) + self.rating_means[user_id]
        mask = self.mat_mask[:, user_id]
        y_pred = y_pred[mask]
        if bound is not None:
            y_pred = np.clip(y_pred, bound[0], bound[1])
        return y_pred

    def evaluate_RMSE(self) -> float:
        dot_product = self.X.dot(self.W)
        if self.user_based:
            bias = self.rating_means.reshape(-1, 1)
        else:
            bias = self.rating_means
        delta = self.raw_data - (dot_product + bias)
        delta = np.ma.array(delta, mask=self.mat_mask)
        RMSE = np.sqrt(np.mean(delta**2))

        return RMSE
