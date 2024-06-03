import numpy as np
import pandas as pd
import random
import itertools
import time

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

from config import MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_INFORMATION_GAIN, TEST_SIZE, RANDOM_STATE, DATA_PATH, \
    MAX_CATEGORIES


class MyDecisionTreeRegressor:

    def __init__(self, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                 min_information_gain=MIN_INFORMATION_GAIN, max_categories=MAX_CATEGORIES):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.max_categories = max_categories
        self.tree = None

    @staticmethod
    def _variance(s):
        return 0 if len(s) == 1 else s.var()

    @staticmethod
    def _get_information_gain(y, condition):

        left_split_size = condition.sum()
        right_split_size = len(condition) - left_split_size

        if left_split_size == 0 or right_split_size == 0:
            return 0

        total_size = len(y)
        left_weight = left_split_size / total_size
        right_weight = right_split_size / total_size

        if y.dtype != 'O':
            return (
                    MyDecisionTreeRegressor._variance(y) -
                    left_weight * MyDecisionTreeRegressor._variance(y[condition]) -
                    right_weight * MyDecisionTreeRegressor._variance(y[~condition])
            )
        else:
            raise Exception("Target is not a numerical data")

    @staticmethod
    def _get_conditions(s: pd.Series):
        if s.dtypes != 'O':
            raise ValueError("The provided series is not categorical")

        unique_values = list(s.unique())
        conditions = [list(subset) for L in range(len(unique_values))
                      for subset in itertools.combinations(unique_values, L)]

        return conditions[1:-1]

    @staticmethod
    def get_midpoints(s: pd.Series):
        sorted_unique_values = np.sort(s.unique())
        midpoints = (sorted_unique_values[:-1] + sorted_unique_values[1:]) / 2
        return midpoints

    @staticmethod
    def _best_split_for_column(x: pd.Series, y: pd.Series):
        split_value = []
        information_gains = []

        is_categorical = x.dtypes == 'O'

        if not is_categorical:
            conditions = MyDecisionTreeRegressor.get_midpoints(x)
        # else:
        #     conditions = MyDecisionTreeRegressor._get_conditions(x)

        x_values = x.values
        y_values = y.values

        for condition in conditions:
            condition_mask = x_values < condition if not is_categorical else np.isin(x_values, condition)
            information_gain = MyDecisionTreeRegressor._get_information_gain(y_values, condition_mask)
            information_gains.append(information_gain)
            split_value.append(condition)

        if len(information_gains) != 0:
            best_ig = max(information_gains)
            best_ig_index = information_gains.index(best_ig)
            best_split = split_value[best_ig_index]
            return best_ig, best_split, not is_categorical, True
        else:
            return None, None, None, False

    @staticmethod
    def _get_best_split(df: pd.DataFrame, y: pd.Series):
        masks = df.apply(MyDecisionTreeRegressor._best_split_for_column, args=(y,))

        if not masks.loc[3, :].any():
            return None, None, None, None

        valid_masks = masks.columns[masks.loc[3, :].astype(bool)]
        valid_results = masks[valid_masks]

        # проверяем что все information gains валидны и флоты
        ig_series = valid_results.loc[0].astype(float)

        split_feature = ig_series.idxmax()
        split_value = valid_results.loc[1, split_feature]
        information_gain = valid_results.loc[0, split_feature]
        is_numeric = valid_results.loc[2, split_feature]

        return split_feature, split_value, information_gain, is_numeric

    @staticmethod
    def _make_split(variable, value, data, is_numeric):
        if is_numeric:
            mask = data[variable] < value
        else:
            mask = data[variable].isin(value)

        left_leaf_data = data[mask]
        right_leaf_data = data[~mask]

        return left_leaf_data, right_leaf_data

    def fit(self, X, y):
        X[y.name] = y
        self.tree = self._fit_tree_regressor(X, y.name, self.max_depth, self.min_samples_split,
                                             self.min_information_gain, 0, self.max_categories)

    def _fit_tree_regressor(self, data, y, max_depth=None, min_samples_split=None,
                            min_information_gain=MIN_INFORMATION_GAIN,
                            depth_counter=0,
                            max_categories=20):
        if depth_counter == 0:
            check_columns = data.select_dtypes(include=["object"]).columns
            for column in check_columns:
                var_length = len(data[column].unique())
                if var_length > max_categories:
                    raise ValueError(
                        f'{column} exceeds the limit of unique values {max_categories}')

        depth_cond = (max_depth is None) or (depth_counter < max_depth)
        sample_cond = (min_samples_split is None) or (data.shape[0] > min_samples_split)

        if depth_cond and sample_cond:
            feature_name, split_value, information_gain, feature_type = MyDecisionTreeRegressor._get_best_split(
                data.drop(y, axis=1), data[y])

            if information_gain is not None and information_gain >= min_information_gain:
                depth_counter += 1
                left, right = MyDecisionTreeRegressor._make_split(feature_name, split_value, data, feature_type)

                split_type = "<=" if feature_type else "in"
                question = f"{feature_name} {split_type} {split_value}"
                subtree = {question: []}

                no_answer = self._fit_tree_regressor(right, y, max_depth, min_samples_split, min_information_gain,
                                                     depth_counter)
                yes_answer = self._fit_tree_regressor(left, y, max_depth, min_samples_split, min_information_gain,
                                                      depth_counter)

                if yes_answer == no_answer:
                    return yes_answer
                else:
                    subtree[question] = [yes_answer, no_answer]
                return subtree
        return np.mean(data[y])
        # return np.median(data[y])

    def predict(self, observations):
        my_predictions = []

        for idx, observation in observations.iterrows():
            pred = self._predict(observation, self.tree)
            my_predictions.append(pred)

        return my_predictions

    def _predict(self, observation, tree):
        condition = list(tree.keys())[0]
        feature, operator, value = condition.split(' ', 2)

        if operator == '<=':
            ans = tree[condition][0] if observation[feature] <= float(value) else tree[condition][1]
        else:
            ans = tree[condition][0] if feature in value else tree[condition][1]

        return self._predict(observation, ans) if isinstance(ans, dict) else ans


if __name__ == '__main__':

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(' ', '_')
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    df_wo_nulls = df[~df.isnull().any(axis=1)]

    X, Y = df_wo_nulls.drop(['Hotel_Name', 'Price', 'Description'], axis=1), np.log(df_wo_nulls['Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    regr = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                                 min_impurity_decrease=MIN_INFORMATION_GAIN)
    regr.fit(X_train, y_train)

    preds = regr.predict(X_train)
    print("Sklearn tree")
    print("Train:")
    print("MAE:", mean_absolute_error(np.exp(y_train), np.exp(preds)))
    print("MAPE:", mean_absolute_percentage_error(np.exp(y_train), np.exp(preds)))

    preds = regr.predict(X_test)
    print("Test:")
    print("MAE:", mean_absolute_error(np.exp(y_test), np.exp(preds)))
    print("MAPE:", mean_absolute_percentage_error(np.exp(y_test), np.exp(preds)))

    plt.figure(figsize=(20, 10))
    plot_tree(regr, filled=True, feature_names=X_train.columns, rounded=True, fontsize=10)
    plt.show()

    start_time = time.time()
    tree = MyDecisionTreeRegressor(MAX_DEPTH, MIN_SAMPLES_SPLIT,
                                   MIN_INFORMATION_GAIN)
    tree.fit(X_train, y_train)
    # print(categorical_options(X['Бар']))
    my_predictions = tree.predict(X_train)
    print("My tree")
    print("Train:")
    print("MAE:", mean_absolute_error(np.exp(y_train), np.exp(my_predictions)))
    print("MAPE:", mean_absolute_percentage_error(np.exp(y_train), np.exp(my_predictions)))

    my_predictions = tree.predict(X_test)
    print(my_predictions)
    print("Test:")
    print("MAE:", mean_absolute_error(np.exp(y_test), np.exp(my_predictions)))
    print("MAPE:", mean_absolute_percentage_error(np.exp(y_test), np.exp(my_predictions)))
    end_time = time.time()
    print("tree", tree.tree)

    print(f"My tree execution time: {end_time - start_time} seconds")
