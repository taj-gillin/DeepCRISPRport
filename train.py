import os
from preprocess import get_ae_data, get_on_target_data, get_on_target_reg_data, get_off_target_data, get_off_target_reg_data
from models import *
from utils import debug_print

def train_ae():
    # Load data
    debug_print(["Training autoencoder..."])
    x_train, y_train, x_test, y_test = get_ae_data()
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = DCDNN()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    model.save_weights('./models/ae_weights.keras')

def train_on_target():
    debug_print(["Training on-target model..."])
    x_train, y_train, x_test, y_test = get_on_target_data()
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = OnTarget()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    model.save_weights('./models/on_target_weights.keras')

def train_on_target_reg():
    debug_print(["Training on-target reg model..."])
    x_train, y_train, x_test, y_test = get_on_target_reg_data()
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = OnTargetReg()
    model.compile(optimizer='adam', loss='mse')

    # Train model
    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    model.save_weights('./models/on_target_reg_weights.keras')


def train_off_target():
    debug_print(["Training off-target model..."])
    x_train, y_train, x_test, y_test = get_off_target_data()
    debug_print(["Data loaded. Test size: ", x_train[0].shape[0], ", Train size: ", x_test[0].shape[0]])
    # Init model
    model = OffTarget()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    model.save_weights('./models/off_target_weights.keras')

def train_off_target_reg():
    debug_print(["Training off-target reg model..."])
    x_train, y_train, x_test, y_test = get_off_target_reg_data()
    debug_print(["Data loaded. Test size: ", x_train[0].shape[0], ", Train size: ", x_test[0].shape[0]])

    # Init model
    model = OffTargetReg()
    model.compile(optimizer='adam', loss='mse')

    # Train model
    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    model.save_weights('./models/off_target_reg_weights.keras')
    

if __name__ == '__main__':
    os.system('clear')
    debug_print("DEEP CRISPR")
    train_ae()
    train_on_target()
    train_on_target_reg()
    train_off_target()
    train_off_target_reg()
    