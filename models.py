import tensorflow as tf
import os
from utils import debug_print

# Assuming each sgRNA sequence is 23 nucleotides long and we have 4 channels for nucleotides and 4 for epigenomic data
input_shape = (23, 8)  # 23 nucleotides, 8 channels (4 one-hot encoded bases + 4 epigenomic features)

class DCDNN(tf.keras.Model):  # Deep Convolutionary Denoising Neural Network
    def __init__(self):
        super(DCDNN, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(64, 3, strides=1, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1DTranspose(32, 3, strides=1, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(8, 3, activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    

class OnTarget(tf.keras.Model):
    def __init__(self, encoder_file = './models/ae_weights.keras'):
        super(OnTarget, self).__init__()
       
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 3, activation='sigmoid', padding='same')
        ])

        # Load encoder weights if applicable
        # TODO: make less scuffed
        if os.path.exists(encoder_file):
            ae = DCDNN()
            ae.compile(optimizer='adam', loss='binary_crossentropy')
            self.encoder.set_weights(ae.encoder.get_weights())
            debug_print(["Loaded encoder weights from ", encoder_file])
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        classified = self.classifier(encoded)
        return classified


class OnTargetReg(tf.keras.Model):
    def __init__(self, encoder_file = './models/ae_weights.keras'):
        super(OnTargetReg, self).__init__()
       
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 3, activation='linear', padding='same')
        ])

        # Load encoder weights if applicable
        # TODO: make less scuffed
        if os.path.exists(encoder_file):
            ae = DCDNN()
            ae.compile(optimizer='adam', loss='binary_crossentropy')
            self.encoder.set_weights(ae.encoder.get_weights())
            debug_print(["Loaded encoder weights from ", encoder_file])


    def call(self, inputs):
        encoded = self.encoder(inputs)
        classified = self.classifier(encoded)
        return classified    
    

class OffTarget(tf.keras.Model):
    def __init__(self, encoder_file = './models/ae_weights.keras'):
        super(OffTarget, self).__init__()

        # Target encoder
        self.target_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Off-target encpder
        self.off_target_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 3, activation='sigmoid', padding='same')
        ])

        # Load encoder weights if applicable
        # TODO: make less scuffed
        if os.path.exists(encoder_file):
            ae = DCDNN()
            ae.compile(optimizer='adam', loss='binary_crossentropy')
            self.target_encoder.set_weights(ae.encoder.get_weights())
            self.off_target_encoder.set_weights(ae.encoder.get_weights())
            debug_print(["Loaded encoder weights from ", encoder_file])

    def call(self, inputs):
        target_encoded = self.target_encoder(inputs[0])
        off_target_encoded = self.off_target_encoder(inputs[1])
        encoded = tf.keras.layers.concatenate([target_encoded, off_target_encoded], axis=1)
        classified = self.classifier(encoded)
        return classified
    

class OffTargetReg(tf.keras.Model):
    def __init__(self, encoder_file = './models/ae_weights.keras'):
        super(OffTargetReg, self).__init__()

        # Target encoder
        self.target_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Off-target encpder
        self.off_target_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(256, 3, activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(1, 3, activation='linear', padding='same')
        ])

        # Load encoder weights if applicable
        # TODO: make less scuffed
        if os.path.exists(encoder_file):
            ae = DCDNN()
            ae.compile(optimizer='adam', loss='binary_crossentropy')
            self.target_encoder.set_weights(ae.encoder.get_weights())
            self.off_target_encoder.set_weights(ae.encoder.get_weights())
            debug_print(["Loaded encoder weights from ", encoder_file])

    def call(self, inputs):
        target_encoded = self.target_encoder(inputs[0])
        off_target_encoded = self.off_target_encoder(inputs[1])
        encoded = tf.keras.layers.concatenate([target_encoded, off_target_encoded], axis=1)
        classified = self.classifier(encoded)
        return classified