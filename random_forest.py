import numpy as np
import pandas as pd
import random


def get_bootstrap(X_train, y_train):
    n = len(X_train)
    bootstrap_indices = np.random.choice(n, n, replace=True)
    oob_indices = list(set(range(n)) - set(bootstrap_indices))

    X_bootstrap = X_train.take(bootstrap_indices)
    y_bootstrap = y_train.take(bootstrap_indices)
    X_oob = X_train.take(oob_indices)
    y_oob = y_train.take(oob_indices)

    return X_bootstrap, y_bootstrap, X_oob, y_oob

