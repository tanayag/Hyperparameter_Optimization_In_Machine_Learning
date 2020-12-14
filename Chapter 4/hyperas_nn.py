from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np

from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    # MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    classes = 10
    input_shape = 784
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)
    return x_train, y_train, x_test, y_test, input_shape, classes


def create_model(x_train, y_train, x_test, y_test, input_shape, classes):
 
    model = Sequential()
    model.add(Dense(units={{choice([8, 16])}}, 
                      input_shape=(input_shape, ),name='dense1'))

    layers = {{choice([2, 3, 4, 5, 6, 7, 8, 9, 10])}}

    for i in range(layers):        
        model.add(Dense(units={{choice([32, 64, 256, 512, 1024])}}))
        model.add(Dropout({{choice([0, 0.33])}}))
        model.add(Activation(activation={{choice(['relu', 'elu'])}}))

    model.add(Dense(classes))
    model.add(Activation(activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                   metrics=['accuracy'],
                    optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
                      batch_size={{choice([4, 8, 16])}},
                      epochs=10,
                      verbose=3,
                      validation_split=0.2)

    validation_acc = np.amax(result.history['val_accuracy'])
    print('Test accuracy:', validation_acc)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': 
                                                                 model}


best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trials=Trials())

X_train, Y_train, X_test, Y_test, _, _ = data()
print("Test Score on Best Model:")
print(best_model.evaluate(X_test, Y_test))
print("Hyperparameter Set for best Model:")
print(best_run)