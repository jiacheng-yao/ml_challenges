from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.metrics import mean
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')

import tempfile
import itertools

from categorical_embedding_tf_model import build_estimator

LABEL_COLUMN = 'saleslog'
CATEGORICAL_COLUMNS = ["country", "article", "promo1", "promo2", "productgroup",
                       "category", "style", "sizes", "gender"]
CONTINUOUS_COLUMNS = ["regular_price", "current_price", "ratio", "cost",
                      "day", "week", "month", "year", "dayofyear",
                      "rgb_r_main_col", "rgb_g_main_col", "rgb_b_main_col",
                      "rgb_r_sec_col", "rgb_g_sec_col", "rgb_b_sec_col"]


def loaddata():
    sales = pd.read_csv('sales.txt',
                        delimiter=';',
                        nrows=None,
                        parse_dates=['retailweek'],
                        date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')))

    attributes = pd.read_csv('article_attributes.txt', delimiter=';')


    sales['saleslog'] = np.log1p(sales['sales'])
    sales['saleslog'] = sales['saleslog'].astype(np.float32)

    features_x = ['regular_price', 'current_price', 'ratio', 'promo1', 'promo2']

    var_name = 'retailweek'

    sales['day'] = pd.Index(sales[var_name]).day
    sales['week'] = pd.Index(sales[var_name]).week
    sales['month'] = pd.Index(sales[var_name]).month
    sales['year'] = pd.Index(sales[var_name]).year
    sales['dayofyear'] = pd.Index(sales[var_name]).dayofyear

    sales['day'] = sales['day'].fillna(0)
    sales['week'] = sales['week'].fillna(0)
    sales['month'] = sales['month'].fillna(0)
    sales['year'] = sales['year'].fillna(0)
    sales['dayofyear'] = sales['dayofyear'].fillna(0)

    features_x.append('day')
    features_x.append('week')
    features_x.append('month')
    features_x.append('year')
    features_x.append('dayofyear')

    # join the two dataframes
    data_merged = pd.merge(sales, attributes, on='article')

    for col in list(data_merged.select_dtypes(include=['object']).columns):
        #     data_merged[col+'_numeric'] = pd.Categorical(data_merged[col])
        #     data_merged[col+'_numeric'] = data_merged[col+'_numeric'].cat.codes
        #     features_x.append(col+'_numeric')
        features_x.append(col)

    for col in ['cost',
                'rgb_r_main_col', 'rgb_g_main_col', 'rgb_b_main_col',
                'rgb_r_sec_col', 'rgb_g_sec_col', 'rgb_b_sec_col']:
        features_x.append(col)

    data_merged.promo1 = data_merged.promo1.astype(str)
    data_merged.promo2 = data_merged.promo2.astype(str)

    df_train = data_merged[data_merged['retailweek'] < '2017-03-31']
    df_test = data_merged[data_merged['retailweek'] > '2017-03-31']
    return df_train, df_test


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
    for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


# def rmspe(labels, predictions, weights=None,
#                         metrics_collections=None,
#                         updates_collections=None,
#                         name=None):
#     absolute_errors = math_ops.abs(predictions - labels)
#     return mean(absolute_errors, weights, metrics_collections,
#                 updates_collections, name or 'rmspe')


def mape(labels, predictions, weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
    absolute_percentage_errors = math_ops.abs(math_ops.div(math_ops.subtract(labels, predictions), labels))
    return mean(absolute_percentage_errors, weights, metrics_collections,
                updates_collections, name or 'mape')


def streaming_mape(predictions, labels, weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
    return mape(
        predictions=predictions, labels=labels, weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections, name=name)


def main(unused_argv):
    # Load datasets.
    df_train, df_test = loaddata()

    df_validation = df_train[df_train['retailweek'] > '2017-02-28']
    df_train = df_train[df_train['retailweek'] < '2017-02-28']

    model_dir = "/home/jc/workspace/adidas_take_home/model_dir/"
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir)
    # m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    # results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    # for key in sorted(results):
    #     print("%s: %s" % (key, results[key]))

    validation_metrics = {
        "mae":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_mean_absolute_error,
            ),
        "mape":
            tf.contrib.learn.MetricSpec(
                metric_fn=streaming_mape,
            )
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: input_fn(df_validation),
        eval_steps=1,  # Try adding this
        metrics=validation_metrics,
        every_n_steps=50,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=50
    )

    m.fit(
        input_fn=lambda: input_fn(df_train),
        steps=50000,
        monitors=[validation_monitor],
    )

    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    # Print out predictions
    y = m.predict(input_fn=lambda: input_fn(df_test))
    # .predict() returns an iterator; convert to a list and print predictions
    # islice(y,6) only the first 6 prediction results
    predictions = list(itertools.islice(y, 6))
    real = list(df_test[LABEL_COLUMN])[:6]
    print("Predictions: {}".format(str(predictions)))
    print("Real Values: {}".format(str(real)))


if __name__ == "__main__":
    tf.app.run()
