import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE


def get_model_metrics(X, y, model, target_is_log=False, random_state=RANDOM_STATE, verbal=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if target_is_log:
        y_train, y_test = np.exp(y_train), np.exp(y_test)
        y_pred_train, y_pred_test = np.exp(y_pred_train), np.exp(y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    if verbal:
        print("Train:")
        print("MAE:", mae_train)
        print("MAPE:", mape_train)
        print("MSE:", mse_train)

        print("Test:")
        print("MAE:", mae_test)
        print("MAPE:", mape_test)
        print("MSE:", mse_test)

    return mae_train, mape_train, mse_train, mae_test, mape_test, mse_test


def has_keyword(row, keyword, name_feature='Hotel_Name', description_feature='Description'):
    name = row[name_feature].lower() if isinstance(row[name_feature], str) else ""
    description = row[description_feature].lower() if isinstance(row[description_feature], str) else ""
    if name is None or description is None:
        return 0
    return 1 if keyword in name or keyword in description else 0