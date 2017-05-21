import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Slide over and chop of window sized slices
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])

    # Reshape and return
    return np.array(X), np.array(y).reshape(-1, 1)


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # Build an RNN to perform regression on our time series input/output data
    model = Sequential()
    # Add LSTM with 5 hidden layers
    model.add(LSTM(5, input_shape=X_train.shape[1:]))
    # Output
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # remove as many non-english characters and character sequences as you can 
    chars_to_delete = '"$%&\'\\()*-/+@´`^¨àâèé0123456789'

    for l in chars_to_delete:
        text = text.replace(l, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # Slide over and chop of window sized slices
    for i in range(window_size, len(text), step_size):
        inputs.append(text[i - window_size:i])
        outputs.append(text[i])

    return inputs,outputs
