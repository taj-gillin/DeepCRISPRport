import os
from preprocess import *
from models import *
from utils import debug_print

def train_ae(x_train, y_train, x_test, y_test, batch_size = 128, epochs = 10, model_file = './models/ae_weights.keras'):
    # Load data
    debug_print(["Training autoencoder..."])
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = DCDNN()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    debug_print(["Saving model weights."])
    model.save_weights(model_file)

def train_on_target(x_train, y_train, x_test, y_test, encoder_file, batch_size = 128, epochs = 10):
    debug_print(["Training on-target model..."])
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = OnTarget(encoder_file)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    debug_print(["Saving model weights."])
    model.save_weights('./models/on_target_weights.keras')

def train_on_target_reg(x_train, y_train, x_test, y_test, encoder_file, batch_size = 128, epochs = 10):
    debug_print(["Training on-target reg model..."])
    debug_print(["Data loaded. Test size: ", x_train.shape[0], ", Train size: ", x_test.shape[0]])

    # Init model
    model = OnTargetReg(encoder_file)
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    debug_print(["Saving model weights."])
    model.save_weights('./models/on_target_reg_weights.keras')


def train_off_target(x_train, y_train, x_test, y_test, encoder_file, batch_size = 128, epochs = 10):
    debug_print(["Training off-target model..."])
    debug_print(["Data loaded. Test size: ", x_train[0].shape[0], ", Train size: ", x_test[0].shape[0]])

    # Init model
    model = OffTarget(encoder_file)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    debug_print(["Saving model weights."])
    model.save_weights('./models/off_target_weights.keras')

def train_off_target_reg(x_train, y_train, x_test, y_test, encoder_file, batch_size = 128, epochs = 10):
    debug_print(["Training off-target reg model..."])
    debug_print(["Data loaded. Test size: ", x_train[0].shape[0], ", Train size: ", x_test[0].shape[0]])

    # Init model
    model = OffTargetReg(encoder_file)
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    # Save model
    debug_print(["Saving model weights."])
    model.save_weights('./models/off_target_reg_weights.keras')
    

if __name__ == '__main__':
    os.system('clear')
    debug_print("DEEP CRISPR")
    train_ae()
    train_on_target()
    train_on_target_reg()
    train_off_target()
    train_off_target_reg()
    