# Recommendation system

## Environment setup
Require `python=3.9`.

Run `pip install -r requirements.txt` to install dependencies.

## Methodology
This repo implements the [matrix factorization technique](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) with gradient descent for optimization on two different datasets, [Frappe](https://arxiv.org/abs/1505.03014) and [TripAdvisor](https://ieeexplore.ieee.org/document/6927637). 

- For TripAdvisor dataset, the target for prediction is the rating of items. 
- For Frappe dataset, as this dataset does not contain any rating for the apps, I use the number of app usage as target with three different variants as follows:
  - **Frappe**: Predict the number of usage directly.
  - **Frappe-binning**: Divide the number of usages into 10 bins and predict the bins.
  - **Frappe-lognorm**: Obtain the frequency of usage by taking log of the number of usages and normalizing into range [1, 5], then predict the frequency.
- In each dataset, I split 10% to use as the unseen test set and the rest for training. 

To search for optimal hyper-parameters, I employ grid-search on training set with cross-validation over a variety combinations with RMSE as scoring metric. Here are the list of hyper-parameters search space:
```
# K: size of latent
# user_based: Normalize matrix based on users or items
# lr: learning rate for gradient descent
# lam: regularization factor
{
    "k": [2, 5, 10, 100, 1000],  
    "user_based": [True, False],  
    "lr": [0.1, 0.5, 1.0],
    "lam": [0.0, 0.1, 0.5],
}
```
After the best hyper-params combination is found, I refit the model on whole training set and measure RMSE on the unseen test set.

## Run experiments
Run this command to perform a hyper-parameters search and refit with best combination
```
# <dataset> can take one of the following values: [frappe, frappe_binning, frappe_lognorm, tripadvisor]

python run_gridsearch.py --dataset <dataset> --n_jobs -1
```

## Results
Here are the results of best hyper-parameters on unseen test sets. Detailed hyper-parameters search results are found in the `results` folder.

| Dataset | Parameters | RMSE |
|---------|------------|------| 
| TripAdvisor | k=5, user_based=True, lr=1.0, lam=0.5 | 1.0697 |
| Frappe (standard) | k=5, user_based=True, lr=1.0, lam=0.5 | 424.3517 |
| Frappe (binning) | k=100, user_based=True, lr=1.0, lam=0.5 | 2.8048 |
| Frappe (log norm) | k=2, user_based=True, lr=0.1, lam=0.1 | 0.7579 |