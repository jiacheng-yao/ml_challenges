import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "IdentityTransform",
    "FeatureDefinition",
    "get_train_test_indices",
    "train_test_pools_from_dataframe",
    "train_test_sets_from_dataframe",
    "transform_data",
    "transform_data_for_catboost",
    "untransformed_data_from_dataframe",
]


class IdentityTransform:
    """
    An sklearn.Pipeline compatible preprocessor that does nothing.
    """

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if y:
            return X, y
        else:
            return X

    def fit_transform(self, X, y=None):
        if y:
            return X, y
        else:
            return X


class FeatureDefinition:
    """
    FeatureDefinition describes which columns of a dataframe are to be used as features for a supervised machine learning
    algorithm, and which pre-processing is to be applied to those columns.
    """

    def __init__(self, for_catboost: bool = False):
        """
        :param for_catboost:
            Set for_catboost=True if Catboost is your learning algrithm.
        """
        # these lists determine column order!
        self.categorical_columns: List[str] = []
        self.continuous_columns: List[str] = []
        # maps column name : [values]
        self.categorical_values: Dict[str, np.ndarray] = {}
        # maps column name : sklearn.Pipeline
        self.continuous_pipelines: Dict[str, Any] = {}
        self.for_catboost = for_catboost
        # list of categorical columns which were created with fillna=True
        self.fillna_categoricals: Dict[str, str] = {}

    def add_categorical_column(
        self,
        column_name: str,
        data: pd.DataFrame,
        do_fillna: bool = False,
        fillna_value: str = "(na)",
    ):
        """
        Defines column_name as a categorical feature.

        Performs 'fillna(fillna_value)' on the columns of the original data frame if fillna==True.
        Fails if no fillna is performed and the column contain NaNs.
        Computes the unique values of the column if for_catboost=False.
        """
        assert column_name in data.columns, f"Column {column_name} not in dataframe!"
        if do_fillna:
            logger.info(f'Performing "fillna({fillna_value})" on column {column_name}.')
            data[column_name].fillna(fillna_value, inplace=True)
            self.fillna_categoricals[column_name] = fillna_value
        else:
            assert (
                data[column_name].isnull().sum() == 0
            ), f"categorical column {column_name} contains NAs!"
        self.categorical_columns.append(column_name)
        self.categorical_values[column_name] = data[column_name].unique()

    def apply_fillna(self, data: pd.DataFrame, na_value: str = "(na)"):
        for column in self.fillna_categoricals:
            assert column in data.columns
            logger.info(f"Performing fillna in column {column}.")
            data[column] = data[column].fillna(na_value)

    def add_categorical_columns(
        self,
        column_names: List[str],
        data: pd.DataFrame,
        do_fillna=False,
        fillna_value="(na)",
    ):
        """
        Defines multiple columns as a categorical columns and determines the unique values.
        See add_categorical_column for details.
        """
        for name in column_names:
            self.add_categorical_column(name, data, do_fillna, fillna_value)

    def add_continuous_column_with_pipeline(
        self, column_name: str, data: pd.DataFrame, transformation_pipeline: Any
    ):
        """
        Defines column_name as a continuous feature to be preprocessed with the sklearn
        transformer or transformer pipeline specified. Fits the given pipeline to the data and stores
        the fitted pipeline internally. transformation_pipeline should not be changed
        afterwards.

        :param column_name:
        :param data:
        :param transformation_pipeline:
            an sklearn transformer or pipeline or any class compatible with that interface
        """
        assert column_name in data.columns, f"Column {column_name} not in dataframe!"
        assert (
            column_name not in self.categorical_values
        ), f"{column_name} already defined as categorical column"
        assert (
            column_name not in self.continuous_pipelines
        ), f"{column_name} already defined as continuous column"

        if not isinstance(transformation_pipeline, IdentityTransform):
            data_array = data[column_name].values.reshape(-1, 1)
            transformation_pipeline.fit(data_array)
        self.continuous_pipelines[column_name] = transformation_pipeline
        self.continuous_columns.append(column_name)

    def add_continuous_column_standard_scaled(
        self, column_name: str, data: pd.DataFrame
    ):
        """
        Defines column_name as a continuous feature to be pre-preocessed with sklearn.StandardScaler.
        Creates and fits the preprocessor.
        """
        pipeline = Pipeline([("scale", StandardScaler())])
        self.add_continuous_column_with_pipeline(column_name, data, pipeline)

    def add_continuous_column_log_standard_scaled(
        self, column_name: str, data: pd.DataFrame
    ):
        """
        Defines column_name as a continuous feature to be pre-preocessed with using the following pieline:

            Pipeline([('log', FunctionTransformer(np.log1p)),
                      ('scale', StandardScaler())])

        Creates and fits the preprocessor.
        """
        pipeline = Pipeline(
            [("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]
        )
        self.add_continuous_column_with_pipeline(column_name, data, pipeline)

    def add_continuous_column(
        self,
        column_name: str,
        data: pd.DataFrame,
    ):
        """
        Defines column_name as a continuous column without any data transformation.
        """
        self.add_continuous_column_with_pipeline(column_name, data, IdentityTransform())

    def get_continuous_pipeline(self, column_name: str):
        """
        Returns the fitted transformation pipeline for a continuous column, or None if there is
        not pipeline for the column.
        """
        return self.continuous_pipelines.get(column_name, None)

    def get_categorical_feature_names(self):
        if self.for_catboost:
            return self.categorical_columns

        # Otherwise use one hot encoding for the feature names.
        categorical_features = []
        for col in self.categorical_columns:
            categorical_features.extend(
                [f"{col}_{value}" for value in self.categorical_values[col]]
            )
        return categorical_features

    def get_feature_names(self):
        """
        The (column) names of the features.
        :return:
        """
        return self.get_categorical_feature_names() + self.continuous_columns


def transform_data_for_catboost(
    data: pd.DataFrame,
    feature_definition: FeatureDefinition,
    target_column: str,
    weight: Optional[List] = None,
) -> Tuple[Pool, List[str]]:
    """
    Creates a Catboost pool from the specified dataframe using the specified FeatureDefinition.
    :param data:
        the data
    :param feature_definition:
        the FeatureDefinition
    :param target_column:
        the column name of the prediction target
    :param weight:
        Optional weights to apply
    :return:
        (Catboost pool, list of variable names according the FeatureDefinition)
    """
    assert (
        target_column in data.columns
    ), f"Missing target column {target_column} in data."

    columns_continuous = []
    for col_name in feature_definition.continuous_columns:
        pipeline = feature_definition.continuous_pipelines[col_name]
        columns_continuous.append(
            pipeline.transform(data[col_name].values.reshape(-1, 1))
        )

    if feature_definition.categorical_columns:
        to_concat = [data.reset_index()[feature_definition.categorical_columns]]
        if len(columns_continuous) > 0:
            to_concat.append(pd.DataFrame(np.hstack(columns_continuous)))
        df_for_pool = pd.concat(to_concat, axis=1, ignore_index=True)
    else:
        df_for_pool = pd.concat(
            [pd.DataFrame(np.hstack(columns_continuous))], axis=1, ignore_index=True
        )

    names = (
        feature_definition.categorical_columns + feature_definition.continuous_columns
    )
    df_for_pool.columns = names

    n_categorical = len(feature_definition.categorical_columns)

    pool = Pool(
        data=df_for_pool,
        label=data[target_column],
        cat_features=np.arange(n_categorical),
        weight=weight,
    )

    return pool, names


def untransformed_data_from_dataframe(
    data: pd.DataFrame,
    feature_definition: FeatureDefinition,
    row_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    if row_indices is not None:
        return data[
            feature_definition.categorical_columns
            + feature_definition.continuous_columns
        ].iloc[row_indices]
    else:
        return data[
            feature_definition.categorical_columns
            + feature_definition.continuous_columns
        ]


def transform_data(data: pd.DataFrame, feature_definition: FeatureDefinition):
    """
    Transforms data according to the given FeatureDefinition, using a matrix (np.array) as output format.

    :return: A dense matrix X (np.array) to use with machine learning algorithms
    and a list of column names for the matrix to reconstruct the origins of the columns X in data.
    """
    # horizontal blocks of the X matrix
    X_blocks = []
    names = []

    if feature_definition.categorical_columns:
        X_cat_df = pd.get_dummies(
            pd.DataFrame(
                [
                    pd.Categorical(
                        data[col], categories=feature_definition.categorical_values[col]
                    )
                    for col in feature_definition.categorical_columns
                ],
                columns=feature_definition.categorical_values,
            )
        )
        X_blocks.append(X_cat_df.values)
        names += list(X_cat_df.columns)

    for col_name in feature_definition.continuous_columns:
        pipeline = feature_definition.continuous_pipelines[col_name]
        transformed = pipeline.transform(data[col_name].values.reshape(-1, 1))
        X_blocks.append(transformed)
        names.append(col_name)

    return np.hstack(X_blocks), names


def get_train_test_indices(
    data: pd.DataFrame,
    test_fraction: float,
    cutoff_or_random="random",
    seed=None,
    cutoff_column=None,
    cutoff_value=None,
):
    """
    Generate indices for a simple train test split of the given dataframe.
    :param data:
        Dataframe to get train and test indices from
    :param test_fraction:
        test set size is int(test_fraction * len(data))
    :param cutoff_or_random:
        if "random":
            randomly selected indices
        if "cutoff" and cutoff_column==cutoff_value==None:
            cuts the dataframe at int(len(data) * test_fraction); first part is used as training set
        if "cutoff" and cutoff_column and cutoff_value are provided:
            training set is data[flt] with flt = data[cutoff_column] <= cutoff_value
    :param cutoff_value: see cutoff_or_random
    :param cutoff_column: see cutoff_or_random
    :param seed: seed for the random number generator

    :return: (array of test indices, array of training indices) into the given dataframe; to be used with iloc
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    indices = np.arange(n)
    n_train = int(n * (1 - test_fraction))

    if cutoff_or_random == "random":
        logger.info("Random train/test split.")
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    elif cutoff_or_random == "cutoff":
        if cutoff_column is None:
            logger.info(
                "Train/test split via cutoff by proportion; make sure the data is sorted!"
            )
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
        else:
            logger.info(
                f"Train/test split via cutoff using column {cutoff_column} at {cutoff_value}."
            )
            train_flt = data[cutoff_column] <= cutoff_value
            train_indices = list(data.index[train_flt])
            test_indices = list(data.index[~train_flt])
    elif "_fold_" in cutoff_or_random:
        k = int(cutoff_or_random.split("_fold_")[0])
        fold_nr = int(cutoff_or_random.split("_fold_")[1])

        fold_size = int(np.ceil(n / k))

        start_index = fold_nr * fold_size
        end_index = min(
            (fold_nr + 1) * fold_size, n + 1
        )  # use at maximum the length of the dataset as end_index

        test_indices = indices[start_index:end_index]
        # train_indices = [index for index in indices if index not in test_indices]
        train_indices = indices[
            :start_index
        ]  # only train on the dataset that occurred prior to the test dataset to not train a model on future data
        logger.info(f"start_end: {start_index}, {end_index}")
    else:
        raise ValueError(
            f"Wrong value for 'cutoff_or_random' parameter provided: {cutoff_or_random}"
        )

    return train_indices, test_indices


def train_test_sets_from_dataframe(
    data: pd.DataFrame,
    target_column: str,
    test_fraction: float,
    feature_def: FeatureDefinition,
    seed=None,
    return_indices=False,
):
    assert target_column in data.columns

    ind_train, ind_test = get_train_test_indices(data, test_fraction, seed=seed)

    X_train, names = transform_data(data.iloc[ind_train], feature_def)
    y_train = data[target_column].iloc[ind_train]

    if test_fraction != 0.0:
        X_test, names2 = transform_data(data.iloc[ind_test], feature_def)
        y_test = data[target_column].iloc[ind_test]
        assert names == names2

    else:
        X_test = None
        y_test = None

    if return_indices:
        return X_train, y_train, X_test, y_test, names, ind_train, ind_test
    else:
        return X_train, y_train, X_test, y_test, names


def train_test_pools_from_dataframe(
    data: pd.DataFrame,
    target_column: str,
    test_fraction: float,
    feature_def: FeatureDefinition,
    return_train_test_indices=False,
    cutoff_or_random="random",
    cutoff_column=None,
    cutoff_value=None,
    seed=None,
    weight=None,
):
    """
    Takes the data in the given dataframe, and the given FeatureDefinition to create Pool data structures to use with Catboost.

    The data is split into training and test set using get_train_test_indices. See documentation there for how to control the split.
    By default, data is split randomly.

    :param data:
    :param target_column:
    :param test_fraction:
        test set size is int(test_fraction * len(data))
    :param feature_def:
        definition of features; must be consistent with the provided dataframe
    :param return_train_test_indices:
        if True, the indices of the samples used for the test and training set are returned
    :param cutoff_or_random:
        see get_train_test_indices
    :param cutoff_column:
        see get_train_test_indices
    :param cutoff_value:
        see get_train_test_indices
    :param seed:
        see get_train_test_indices
    :param weight:
        Optional weights to apply
    :return:
        (pool_train, pool_test, names, ind_train, ind_test) if return_train_test_indices=True, where
            pool_train, pool_test: Catboost pools of training and test data
            names: feature names according to the FeatureDefintion, in the order used in the created pools
            ind_train, ind_test: lists of integer indices of the training and test samples
        ind_train, ind_test are not returned if return_train_test_indices=False
    """
    assert target_column in data.columns, f"Missing target column {target_column}!"

    ind_train, ind_test = get_train_test_indices(
        data,
        test_fraction,
        cutoff_or_random,
        seed=seed,
        cutoff_column=cutoff_column,
        cutoff_value=cutoff_value,
    )

    if weight is not None:
        weight_train = weight.iloc[ind_train]
    else:
        weight_train = None

    pool_train, names = transform_data_for_catboost(
        data.iloc[ind_train], feature_def, target_column, weight=weight_train
    )

    if test_fraction != 0.0:

        if weight is not None:
            weight_test = weight.iloc[ind_test]
        else:
            weight_test = None

        pool_test, names2 = transform_data_for_catboost(
            data.iloc[ind_test], feature_def, target_column, weight=weight_test
        )
        assert names == names2
    else:
        pool_test = None

    if return_train_test_indices:
        return pool_train, pool_test, names, ind_train, ind_test
    else:
        return pool_train, pool_test, names


if __name__ == "__main__":
    cat_A = list("abcabcabca")
    cat_B = list("uvwxuvwxuv")
    cont_C = np.arange(len(cat_A)).astype(float)
    cont_D = np.random.uniform(size=len(cat_A))

    df = pd.DataFrame(
        [
            pd.Series(cat_A, name="A"),
            pd.Series(cat_B, name="B"),
            pd.Series(cont_C, name="C"),
            pd.Series(cont_D, name="D"),
            pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="t"),
        ]
    ).transpose()

    print(df)

    features = FeatureDefinition()
    features.add_categorical_columns(["A", "B"], df)
    features.add_continuous_column_standard_scaled("D", df)
    features.add_continuous_column("C", df)

    X, names = transform_data(df, features)
    print(X)
    print(X.sum(axis=0))
    print(X.var(axis=0))
    print(names)
