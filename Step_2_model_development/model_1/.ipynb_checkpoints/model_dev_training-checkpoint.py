# TensorFlow warning supress

import os

##### Comment if not CUDA available ####
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#######################################

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Input

#### Loading train_test dataset ####
path = os.path.abspath('../..')

X_train = np.load(path+'/Step_1_train_test_data/X_y_feature_label/model_1/X_train_M1.npy')
X_test = np.load(path+'/Step_1_train_test_data/X_y_feature_label/model_1/X_test_M1.npy')
y_train = np.load(path+'/Step_1_train_test_data/X_y_feature_label/model_1/y_train_M1.npy')
y_test = np.load(path+'/Step_1_train_test_data/X_y_feature_label/model_1/y_test_M1.npy')

#### Model Architecture ####
model = Sequential([
  
                    Input(shape=(None, *X_train.shape[2:])),
                                      
                    ConvLSTM2D(
                              filters=64,
                              kernel_size=(5, 5),
                              padding="same",
                              return_sequences=True,
                              activation="relu",
                              dropout=0.025,
                              ),
                    
                    BatchNormalization(),
                  
                    ConvLSTM2D(
                              filters=32,
                              kernel_size=(3, 3),
                              padding="same",
                              return_sequences=True,
                              activation="tanh",
                              dropout=0.02,
                              ),
                  
                    BatchNormalization(),
                  
                    ConvLSTM2D(
                              filters=16,
                              kernel_size=(1, 1),
                              padding="same",
                              return_sequences=True,
                              activation="relu",
                              dropout=0.005,
                              ),
                    
                    Conv3D(
                          filters=1, 
                          kernel_size=(3, 3, 3), 
                          activation="sigmoid", 
                          padding="same"
                          ),
                    
                    ])

#### Compiling Model ####
model.compile(
  loss=keras.losses.binary_crossentropy, 
  optimizer=keras.optimizers.Adam(learning_rate=0.001,),
)

#### Early stopping and adjusting learning rate ####
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)

#### Model checkpoint and its settings ####
checkpoint_filepath = path+"/Step_2_model_development/model_1/checkpoints/"


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
#### Model Fitting ####
epochs = 25
batch_size = 15

history = model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[model_checkpoint_callback, early_stopping, reduce_lr],
                    )

#### Saving trained model ####
model.save(path+'/Step_2_model_development/model_1/trained_model_1.h5')
np.save(path+'/Step_2_model_development/model_1/model_1_history.npy', history.history)