from typing import Optional, Tuple

import numpy as np
import pandas as pd


def rmse(eval_df: pd.DataFrame, model, bound: Optional[Tuple[int, int]]) -> float:
    """Compute RMSE for the given dataset and model."""
    # Compute predictions
    predictions = []
    for _, row in eval_df.iterrows():
        pred = model.predict(row["user"], row["item"], bound=bound)
        predictions.append(pred)
    predictions = np.array(predictions)
    # Compute RMSE
    target = eval_df["target"].values.astype(int)
    # print(predictions, target)
    mse = np.mean((predictions - target) ** 2)
    return np.sqrt(mse)
