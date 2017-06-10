import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from utils import load_data, create_dataset, save


# train the lstm model (including the model definition etc.)
def lstm_train(dataset, application_id):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    # reshape into X=t and Y=t+1
    look_back = 20
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    for i in range(100):
        model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate mean absolute error
    trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])
    print('Train Score: %.2f MAE' % (trainScore))
    testScore = mean_absolute_error(testY[0], testPredict[:,0])
    print('Test Score: %.2f MAE' % (testScore))

    # save the model and the weights
    lstm_save(model, 'model_{}.json'.format(application_id), 'model_weight_{}.h5'.format(application_id))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict[:, 0]
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = testPredict[:, 0]
    # plot baseline and predictions
    plt.title('LSTM Prediction for Ranking (ID = {})'.format(application_id))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)

    # plt.legend(['Real', 'Prediction(Train)', 'Prediction(Test)'], ncol=4, loc='upper center',
    #            bbox_to_anchor=[0.5, 1.1],
    #            columnspacing=1.0, labelspacing=0.0,
    #            handletextpad=0.0, handlelength=1.5,
    #            fancybox=True, shadow=True)
    # plt.show()
    save("lstm_ranking_prediction_result_{}".format(application_id), ext="pdf", close=True, verbose=True)


# save the lstm model for individual application
def lstm_save(model, model_file_json = "model.json", model_weight_h5 = "model.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weight_h5)
    print("Saved model to disk")


# load the lstm model for individual application
def lstm_load(model_file_json = "model.json", model_weight_h5 = "model.h5"):
    # load json and create model
    json_file = open(model_file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weight_h5)
    print("Loaded model from disk")


if __name__ == "__main__":
    # fix random seed for reproducibility
    numpy.random.seed(7)

    ranks = pd.read_csv('itunes_application_ranks.csv')
    itn_app = pd.read_csv('itunes_applications.csv')

    TIMESTEPS = 10

    test_application_id = 504575083

    # load the dataset
    rawdata = load_data(ranks, test_application_id)

    dataset = rawdata['rank'].values
    dataset = dataset.astype('float32')

    lstm_train(dataset, test_application_id)
    # import ipdb; ipdb.set_trace()
