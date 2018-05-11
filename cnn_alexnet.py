import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint


path="../Karst/Affectnet_Final_Data/Full/"
X=np.load(path+"Affectnet-96X_Gray-Full.npy")
Y=np.load(path+"Affectnet-Y-Full.npy")
#Y_sm=np.load(path+"Affectnet-Y-Full-sm.npy")

X=X.reshape(X.shape[0],96,96,1)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
image_size=256

enc = OneHotEncoder()
def top2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
Y_train =enc.fit_transform(Y_train.reshape(Y_train.shape[0],1)).toarray()
Y_test = enc.fit_transform(Y_test.reshape(Y_test.shape[0],1)).toarray()

X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255


#path to save model
model_path = 'Alexnet-FullData-gray-Run1.h5'

# Model callbacks
# Referred from https://keras.io/callbacks/

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc',min_delta=0.001, patience=20)
csv_logger = CSVLogger('Alexnet-FullData-gray-Run1-trainingInfo.csv')
model_checkpt= ModelCheckpoint(model_path,monitor='val_acc',save_best_only=True, mode='max',verbose=0)

drout=0.2
output_classes = 7
model = Sequential()

model.add(Conv2D(96,3,strides=(2,2), activation='relu', kernel_initializer="he_normal",input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drout))
model.add(Conv2D(256,3,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drout))
model.add(Conv2D(384,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(drout))
model.add(Conv2D(384,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(drout))
model.add(Conv2D(256,2,strides=(1,1),activation='relu', kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drout))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
#model.add(Dropout(0.3))
model.add(Dense(256, activation = 'relu'))
#model.add(Dropout(0.3))
model.add(Dense(output_classes, activation = 'softmax'))

model.compile(optimizer= 'adam' , loss = 'categorical_crossentropy',
              metrics= ['accuracy',top2_acc])

print (model.summary())			  
model.fit(X_train, Y_train, batch_size=64, nb_epoch=100, validation_split=0.2,callbacks=[lr_reducer, csv_logger,model_checkpt],verbose=2)


score = model.evaluate(X_test, Y_test, verbose=0)

#print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test Top-2 accuracy:', score[2])