import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.utils.np_utils import to_categorical
import utils

sess=tf.Session()
set_session(sess)

path = "Affectnet_data/"
X_train, X_test, Y_train, Y_test = getData(path+"Affectnet-96X-Full.npy", path+"Affectnet-Y-Full.npy", 0.2)

# define path to save model
model_path = 'VGGnet-FullData-batch-Run_interactive_111.h5'
logger_file = 'VGGnet-FullData-batch-Run-trainingInfo_interactive_111.csv'
lr_reducer, early_stopper, csv_logger, model_checkpt = prepareCallbacks(model_path, logger_file)

droprate=0.2
nb_classes = 7

#Build Model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',kernel_initializer="he_normal",padding='same',input_shape=X_train.shape[1:]))
model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #48
model.add(Dropout(droprate))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #24
model.add(Dropout(droprate))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #12
model.add(Dropout(droprate))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #6x6x128
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(droprate))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(droprate))
model.add(Dense(7, activation = 'softmax'))

#Comiple and Fit
model.compile(optimizer = 'adam', 
				loss = 'categorical_crossentropy',
				metrics = ['accuracy',top2_acc])
model.fit(X_train, Y_train, batch_size = 128, nb_epoch = 30, 
				validation_split = 0.2, shuffle = True,
				callbacks = [lr_reducer, csv_logger,model_checkpt],
				verbose = 1)

#Evaluate
scores = model.evaluate(X_test, Y_test, verbose=0)
printMetrics(scores)
print(confusion_matrix(model.predict_classes(X_test), np.argmax(Y_test, axis=1)))
