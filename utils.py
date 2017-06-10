import pandas as pd
import numpy

import os
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.rrule import rrule, DAILY


# Please make sure in the dataframe df the columns 'application_id', 'rank' and 'created_at' exist
# and follow the format in the original Adjust Data Exercise Dataset.
def load_data(df, application_id=479516143, start_date='2016-01-01', end_date='2016-12-31'):
    df_current = df[df['application_id'] == application_id]

    # Some applications don't make into top 1000 for all 366 days in 2016, therefore we need to pad the data
    # before feeding it into the models
    # P.S. here the empty values(which means the app is out of top 1000 at the time) will be filled with 1001
    # to smooth out the time series, which will make the modeling easier
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    dates_in_between = [dt.strftime('%Y-%m-%d') for dt in rrule(DAILY, dtstart=start_date_obj, until=end_date_obj)]

    cols_to_be_deleted = [x for x in df_current.columns if x not in ['created_at', 'application_id', 'rank']]
    df_current.drop(cols_to_be_deleted, axis=1, inplace=True)

    df_current.index = df_current.created_at
    df2 = pd.DataFrame({'created_at': dates_in_between, 'application_id': application_id, 'rank': 1001})
    df2.index = df2.created_at

    df2['rank'] = df_current['rank']
    df2 = df2.fillna(1001)
    return df2.drop('created_at',1).reset_index()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# save a figure from pyplot.
def save(path, ext='png', close=True, verbose=True):
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")


if __name__ == "__main__":
    LOG_DIR = './ops_logs'
    TIMESTEPS = 10
    RNN_LAYERS = [{'steps': TIMESTEPS}]
    DENSE_LAYERS = [10, 10]
    TRAINING_STEPS = 100000
    BATCH_SIZE = 100
    PRINT_STEPS = TRAINING_STEPS / 100

    ranks = pd.read_csv('itunes_application_ranks.csv')
    itn_app = pd.read_csv('itunes_applications.csv')

    TIMESTEPS = 10

    rawdata = load_data(ranks, 951744068)

    # import ipdb; ipdb.set_trace()
