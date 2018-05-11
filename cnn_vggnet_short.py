from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
sess=tf.Session()
set_session(sess)

path = "Affectnet_data/"
X = np.load(path+"Affectnet-96X-Full.npy")
Y = np.load(path+"Affectnet-Y-Full.npy")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train = np.eye(np.unique(Y_train).shape[0])[Y_train]
Y_test = np.eye(np.unique(Y_test).shape[0])[Y_test]
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# Modify the new ground truth labels accordingly
Y_train = np.load('avg_res_train_labels.npy')

def top2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

# define path to save model
model_path = 'VGGnet-FullData-batch-Run_dark_res.h5'
# prepare callbacks
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc',min_delta=0.001, patience=20)
csv_logger = CSVLogger('VGGnet-FullData-batch-Run-trainingInfo_dark_res.csv')
model_checkpt= ModelCheckpoint(model_path,monitor='val_acc',save_best_only=True, mode='max',verbose=0)

droprate=0.2
nb_classes = 7
mean_flag = True

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
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #12
model.add(Dropout(droprate))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #6x6x128
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(droprate))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(droprate))
model.add(Dense(7, activation = 'softmax'))

model.compile(optimizer= 'adam' , loss = 'categorical_crossentropy',metrics= ['accuracy',top2_acc])
model.fit(X_train, Y_train, batch_size=128, nb_epoch=30, validation_split=0.2, shuffle=True, callbacks=[lr_reducer, csv_logger,model_checkpt],verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
#print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test Top-2 accuracy:', score[2])
#print(confusion_matrix(model.predict_classes(X_test), np.argmax(Y_test, axis=1)))
