# TensorFlow warning supress

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Input


path = os.path.abspath("")
X_train = np.load(path+"/train_test_data/X_train.npy") 
X_val = np.load(path+"/train_test_data/X_val.npy") 
y_train = np.load(path+"/train_test_data/y_train.npy") 
y_val = np.load(path+"/train_test_data/y_val.npy") 


# Construct the input layer with no definite frame size.
inp = Input(shape=(None, *X_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = ConvLSTM2D(
                    filters=64,
                    kernel_size=(5, 5),
                    padding="same",
                    return_sequences=True,
                    activation="relu",
                    dropout=0.025,
                )(inp)

x = BatchNormalization()(x)

x = ConvLSTM2D(
                     filters=32,
                     kernel_size=(3, 3),
                     padding="same",
                     return_sequences=True,
                     activation="tanh",
                     dropout=0.02,
                 )(x)

x = BatchNormalization()(x)

x = ConvLSTM2D(
                    filters=16,
                    kernel_size=(1, 1),
                    padding="same",
                    return_sequences=True,
                    activation="relu",
                    dropout=0.005,    
                )(x)

x = Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            )(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001,),
)

keras.callbacks
# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)

# Define modifiable training hyperparameters.
epochs = 25
batch_size = 15


checkpoint_filepath = path+"/model_development/save_checkpoints"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor= 'val_loss',
    verbose = 0,
    save_best_only = False,
    save_weights_only = False,
    mode= 'auto',
    save_freq='epoch',
    options=None,
)


# Fit the model to the training data.
history = model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback, early_stopping, reduce_lr],
                    )

model.save(path+'/predictions/trained_model.h5')
np.save(path+'/predictions/model_history.npy', history.history)

