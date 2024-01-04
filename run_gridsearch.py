import argparse
import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.data_processing import (
    load_tripadvisor,
    load_frappe,
    split_data,
    convert_df_to_utility_mat,
    split_data_cv
)
from src.matrix_factorization import MatrixFactorization


PARAM_GRID = {
    "k": [2, 5, 10, 100, 1000],
    "user_based": [True, False],
    "lr": [0.1, 0.5, 1.0],
    "lam": [0.0, 0.1, 0.5],
    "max_iter": [1000],
    "verbose": [0],
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tripadvisor", "frappe", "frappe_binning"],
        help="Dataset",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of folds for cross validation"
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
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

    PARAM_GRID["random_state"] = [args.seed]
    
    df_train, _, df_test = split_data(df, valid_size=0.0, test_size=0.1, random_state=args.seed)
    split_data_iter = split_data_cv(df_train, cv=args.cv, random_state=args.seed)

    clf = GridSearchCV(
        MatrixFactorization(), param_grid=PARAM_GRID, cv=split_data_iter, verbose=3, n_jobs=args.n_jobs, return_train_score=True
    )
    clf.fit(df)
    with open("results/cv_results.pkl", "wb") as f:
        pickle.dump(clf.cv_results_, f)
    score = clf.score(df_test)
    print(score)


if __name__ == "__main__":
    main()
