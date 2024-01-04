import argparse
import numpy as np

from sklearn.metrics import mean_squared_error

from src.data_processing import (
    load_tripadvisor,
    load_frappe,
    split_data,
    convert_df_to_utility_mat,
)
from src.matrix_factorization import MatrixFactorization

# from src.eval import rmse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tripadvisor", "frappe", "frappe_binning"],
        help="Dataset",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of latent factors")
    parser.add_argument(
        "--user_based",
        action="store_true",
        help="Whether do user-based or item-based CF",
    )
    parser.add_argument(
        "--valid_size", type=float, default=0.1, help="Size of valid set"
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="Size of test set")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.dataset == "tripadvisor":
        df = load_tripadvisor()
        bound = (1, 5)
        # bound = None
    elif args.dataset == "frappe":
        df = load_frappe()
        bound = None
    elif args.dataset == "frappe_binning":
        df = load_frappe(do_binning=True)
        bound = None
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    df_train, df_valid, df_test = split_data(
        df, valid_size=args.valid_size, test_size=args.test_size, random_state=args.seed
    )
    print(f"Size of training set: {df_train.shape[0]}")
    print(f"Size of validation set: {df_valid.shape[0]}")
    print(f"Size of test set: {df_test.shape[0]}")

    mf = MatrixFactorization(
        k=args.k,
        user_based=args.user_based,
        lr=args.lr,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    mf.fit(df_train)

    print(f"Train RMSE: {mf.score(df_train)}")
    print(f"Valid RMSE: {mf.score(df_valid)}")
    print(f"Test RMSE: {mf.score(df_test)}")


if __name__ == "__main__":
    main()
