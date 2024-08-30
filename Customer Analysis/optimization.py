import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
import tensorflow as tf
from keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import joblib
import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def dnn_model(layers = 3, neurons = 32, activation = 'tanh', learning_rate = 0.02, regularization = 0.01, input_shape = (17,)):
    reg = l1_l2(l1=regularization, l2=regularization)
    model = Sequential()
    model.add(Input(input_shape))
    for l in range(layers):
        model.add(Dense(neurons, activation = activation, kernel_regularizer = reg))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return model


scaled_data = joblib.load('scaled_data.sav')
scaled_target = joblib.load('scaled_target.sav')
products = ['Wines','Fruits', 'Meat', 'Fish', 'Sweet', 'Gold']
parameters = {
    'layers':[2,3,4],
    'neurons':[16,32,64,128],
    'activation':['leaky_relu', 'tanh'],
    'regularization':[0.005, 0.01, 0.015],
    'learning_rate':np.arange(0.005, 0.035, 0.01)
}



def optimize_model(i):
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=200,
        min_delta=0.001,
        restore_best_weights=True 
    )
    sklearn_dnn = keras.wrappers.scikit_learn.KerasRegressor(dnn_model)
    random_search = RandomizedSearchCV(sklearn_dnn, param_distributions=parameters, n_iter=30, cv=2)
    random_search.fit(scaled_data, scaled_target[:,i], epochs = 1500, batch_size = 2000, callbacks = [early_stopping], verbose = 0)
    return {products[i]:random_search.best_params_}


if __name__ == '__main__':

    with Pool(6) as p:
        optimized_parameteres = p.map(optimize_model, range(6))
    
    joblib.dump(optimized_parameteres, 'optimized_parameters.joblib')