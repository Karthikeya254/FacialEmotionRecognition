import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras import metrics
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.utils.np_utils import to_categorical=
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from utils import getData, top2_acc, prepareCallbacks, printMetrics

sess=tf.Session()
set_session(sess)

path = "Affectnet_data/"
X_train, X_test, Y_train, Y_test = getData(path+"Affectnet-96X-Full.npy", path+"Affectnet-Y-Full.npy", 0.2)

#path to save model
model_file = 'Final_models/Alexnet.h5'
logger_file = 'Logs/Alexnet_log.csv'
lr_reducer, early_stopper, csv_logger, model_checkpt = prepareCallbacks(model_file, logger_file)

droprate=0.2
nb_classes = 7

#Build Model
model = Sequential()
model.add(Conv2D(96,3,strides=(2,2), activation='relu', kernel_initializer="he_normal",input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(256,3,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))
model.add(Conv2D(384,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(droprate))
model.add(Conv2D(384,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(droprate))
model.add(Conv2D(256,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(output_classes, activation = 'softmax'))

model.compile(optimizer = 'adam',
					loss = 'categorical_crossentropy',
					metrics = ['accuracy',top2_acc])			  
model.fit(X_train, Y_train, batch_size = 64, nb_epoch = 100, validation_split = 0.2,
					callbacks = [lr_reducer, csv_logger,model_checkpt], verbose = 2)

#Evaluate
scores = model.evaluate(X_test, Y_test, verbose=0)
printMetrics(scores)
print(confusion_matrix(model.predict_classes(X_test), np.argmax(Y_test, axis=1)))
