import numpy as np
import tensorflow as tf
import random
import math
import sklearn.metrics.pairwise as skm
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.backend.tensorflow_backend import set_session
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

imgsz = 96
encoding_dim = 500
input_img = Input(shape=(imgsz, imgsz, 3))
num_epochs = 200
bsize = 256

xl1, = np.where(Y_train_lbl == 0)
x0 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x0 = MaxPooling2D((2, 2), padding='same')(x0) #48
x0 = Conv2D(16, (3, 3), activation='relu', padding='same')(x0)
x0 = MaxPooling2D((2, 2), padding='same')(x0) #24
x0 = Conv2D(8, (3, 3), activation='relu', padding='same')(x0)
x0 = MaxPooling2D((2, 2), padding='same')(x0) #12*12*8
x0 = Flatten()(x0)
encoded0 = Dense(encoding_dim, activation='relu')(x0)
x0 = Dense(12*12*8, activation='relu')(encoded0)
x0 = Reshape((12,12,8))(x0)
x0 = Conv2D(8, (3, 3), activation='relu', padding='same')(x0)
x0 = UpSampling2D((2, 2))(x0) #24
x0 = Conv2D(16, (3, 3), activation='relu', padding='same')(x0)
x0 = UpSampling2D((2, 2))(x0) #48
x0 = Conv2D(32, (3, 3), activation='relu', padding='same')(x0)
x0 = UpSampling2D((2, 2))(x0) #96
decoded0 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x0) #96
autoencoder0 = Model(input_img, decoded0)
autoencoder0.compile(optimizer="adam", loss='binary_crossentropy')
encoder0 = Model(input_img, encoded0)
autoencoder0.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)
#===============================================
xl1, = np.where(Y_train_lbl == 1)
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x1 = MaxPooling2D((2, 2), padding='same')(x1) #48
x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2, 2), padding='same')(x1) #24
x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2, 2), padding='same')(x1) #12*12*8
x1 = Flatten()(x1)
encoded1 = Dense(encoding_dim, activation='relu')(x1)
x1 = Dense(12*12*8, activation='relu')(encoded1)
x1 = Reshape((12,12,8))(x1)
x1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
x1 = UpSampling2D((2, 2))(x1) #24
x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x1)
x1 = UpSampling2D((2, 2))(x1) #48
x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
x1 = UpSampling2D((2, 2))(x1) #96
decoded1 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x1) #96
autoencoder1 = Model(input_img, decoded1)
autoencoder1.compile(optimizer="adam", loss='binary_crossentropy')
encoder1 = Model(input_img, encoded1)
autoencoder1.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)

#===============================================
xl1, = np.where(Y_train_lbl == 2)
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x2 = MaxPooling2D((2, 2), padding='same')(x2) #48
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2), padding='same')(x2) #24
x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2), padding='same')(x2) #12*12*8
x2 = Flatten()(x2)
encoded2 = Dense(encoding_dim, activation='relu')(x2)
x2 = Dense(12*12*8, activation='relu')(encoded2)
x2 = Reshape((12,12,8))(x2)
x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x2 = UpSampling2D((2, 2))(x2) #24
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x2 = UpSampling2D((2, 2))(x2) #48
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
x2 = UpSampling2D((2, 2))(x2) #96
decoded2 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x2) #96
autoencoder2 = Model(input_img, decoded2)
autoencoder2.compile(optimizer="adam", loss='binary_crossentropy')
encoder2 = Model(input_img, encoded2)
autoencoder2.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)

#===============================================
xl1, = np.where(Y_train_lbl == 3)
x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x3 = MaxPooling2D((2, 2), padding='same')(x3) #48
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x3)
x3 = MaxPooling2D((2, 2), padding='same')(x3) #24
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x3)
x3 = MaxPooling2D((2, 2), padding='same')(x3) #12*12*8
x3 = Flatten()(x3)
encoded3 = Dense(encoding_dim, activation='relu')(x3)
x3 = Dense(12*12*8, activation='relu')(encoded3)
x3 = Reshape((12,12,8))(x3)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x3)
x3 = UpSampling2D((2, 2))(x3) #24
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x3)
x3 = UpSampling2D((2, 2))(x3) #48
x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
x3 = UpSampling2D((2, 2))(x3) #96
decoded3 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x3) #96
autoencoder3 = Model(input_img, decoded3)
autoencoder3.compile(optimizer="adam", loss='binary_crossentropy')
encoder3 = Model(input_img, encoded3)
autoencoder3.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)

#===============================================
xl1, = np.where(Y_train_lbl == 4)
x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x4 = MaxPooling2D((2, 2), padding='same')(x4) #48
x4 = Conv2D(16, (3, 3), activation='relu', padding='same')(x4)
x4 = MaxPooling2D((2, 2), padding='same')(x4) #24
x4 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
x4 = MaxPooling2D((2, 2), padding='same')(x4) #12*12*8
x4 = Flatten()(x4)
encoded4 = Dense(encoding_dim, activation='relu')(x4)
x4 = Dense(12*12*8, activation='relu')(encoded4)
x4 = Reshape((12,12,8))(x4)
x4 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
x4 = UpSampling2D((2, 2))(x4) #24
x4 = Conv2D(16, (3, 3), activation='relu', padding='same')(x4)
x4 = UpSampling2D((2, 2))(x4) #48
x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x4 = UpSampling2D((2, 2))(x4) #96
decoded4 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x4) #96
autoencoder4 = Model(input_img, decoded4)
autoencoder4.compile(optimizer="adam", loss='binary_crossentropy')
encoder4 = Model(input_img, encoded4)
autoencoder4.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)

#===============================================
xl1, = np.where(Y_train_lbl == 5)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x5 = MaxPooling2D((2, 2), padding='same')(x5) #48
x5 = Conv2D(16, (3, 3), activation='relu', padding='same')(x5)
x5 = MaxPooling2D((2, 2), padding='same')(x5) #24
x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x5)
x5 = MaxPooling2D((2, 2), padding='same')(x5) #12*12*8
x5 = Flatten()(x5)
encoded5 = Dense(encoding_dim, activation='relu')(x5)
x5 = Dense(12*12*8, activation='relu')(encoded5)
x5 = Reshape((12,12,8))(x5)
x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x5)
x5 = UpSampling2D((2, 2))(x5) #24
x5 = Conv2D(16, (3, 3), activation='relu', padding='same')(x5)
x5 = UpSampling2D((2, 2))(x5) #48
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x5)
x5 = UpSampling2D((2, 2))(x5) #96
decoded5 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x5) #96
autoencoder5 = Model(input_img, decoded5)
autoencoder5.compile(optimizer="adam", loss='binary_crossentropy')
encoder5 = Model(input_img, encoded5)
autoencoder5.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)
#===============================================
xl1, = np.where(Y_train_lbl == 6)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #96
x6 = MaxPooling2D((2, 2), padding='same')(x6) #48
x6 = Conv2D(16, (3, 3), activation='relu', padding='same')(x6)
x6 = MaxPooling2D((2, 2), padding='same')(x6) #24
x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(x6)
x6 = MaxPooling2D((2, 2), padding='same')(x6) #12*12*8
x6 = Flatten()(x6)
encoded6 = Dense(encoding_dim, activation='relu')(x6)
x6 = Dense(12*12*8, activation='relu')(encoded6)
x6 = Reshape((12,12,8))(x6)
x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(x6)
x6 = UpSampling2D((2, 2))(x6) #24
x6 = Conv2D(16, (3, 3), activation='relu', padding='same')(x6)
x6 = UpSampling2D((2, 2))(x6) #48
x6 = Conv2D(32, (3, 3), activation='relu', padding='same')(x6)
x6 = UpSampling2D((2, 2))(x6) #96
decoded6 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x6) #96
autoencoder6 = Model(input_img, decoded6)
autoencoder6.compile(optimizer="adam", loss='binary_crossentropy')
encoder6 = Model(input_img, encoded6)
autoencoder6.fit(X_train[xl1], X_train[xl1], epochs=num_epochs, batch_size=bsize, shuffle=True)

#===============================================
x0l1, = np.where(Y_train_lbl == 0)
x1l1, = np.where(Y_train_lbl == 1)
x2l1, = np.where(Y_train_lbl == 2)
x3l1, = np.where(Y_train_lbl == 3)
x4l1, = np.where(Y_train_lbl == 4)
x5l1, = np.where(Y_train_lbl == 5)
x6l1, = np.where(Y_train_lbl == 6)
x0l2, = np.where(Y_test_lbl == 0)
x1l2, = np.where(Y_test_lbl == 1)
x2l2, = np.where(Y_test_lbl == 2)
x3l2, = np.where(Y_test_lbl == 3)
x4l2, = np.where(Y_test_lbl == 4)
x5l2, = np.where(Y_test_lbl == 5)
x6l2, = np.where(Y_test_lbl == 6)
enc_repr0 = np.mean(encoder0.predict(X_train[x0l1]), axis=0)
enc_repr1 = np.mean(encoder1.predict(X_train[x1l1]), axis=0)
enc_repr2 = np.mean(encoder2.predict(X_train[x2l1]), axis=0)
enc_repr3 = np.mean(encoder3.predict(X_train[x3l1]), axis=0)
enc_repr4 = np.mean(encoder4.predict(X_train[x4l1]), axis=0)
enc_repr5 = np.mean(encoder5.predict(X_train[x5l1]), axis=0)
enc_repr6 = np.mean(encoder6.predict(X_train[x6l1]), axis=0)
#===============================================
c1 = 0
c2 = 0
for j in range(len(X_test)):
    xt = X_test[j].reshape((1,imgsz,imgsz,3))
    yt = Y_test_lbl[j]
    enc_test0 = encoder0.predict(xt)
    enc_test1 = encoder1.predict(xt)
    enc_test2 = encoder2.predict(xt)
    enc_test3 = encoder3.predict(xt)
    enc_test4 = encoder4.predict(xt)
    enc_test5 = encoder5.predict(xt)
    enc_test6 = encoder6.predict(xt)
#     e0 = skm.cosine_distances(enc_test0[0].reshape(1,6*6*8), enc_repr0.reshape(1,6*6*8))[0,0]
#     e1 = skm.cosine_distances(enc_test1[0].reshape(1,6*6*8), enc_repr1.reshape(1,6*6*8))[0,0]
#     e2 = skm.cosine_distances(enc_test2[0].reshape(1,6*6*8), enc_repr2.reshape(1,6*6*8))[0,0]
#     e3 = skm.cosine_distances(enc_test3[0].reshape(1,6*6*8), enc_repr3.reshape(1,6*6*8))[0,0]
#     e4 = skm.cosine_distances(enc_test4[0].reshape(1,6*6*8), enc_repr4.reshape(1,6*6*8))[0,0]
#     e5 = skm.cosine_distances(enc_test5[0].reshape(1,6*6*8), enc_repr5.reshape(1,6*6*8))[0,0]
#     e6 = skm.cosine_distances(enc_test6[0].reshape(1,6*6*8), enc_repr6.reshape(1,6*6*8))[0,0]
    e0 = np.mean(np.square(enc_test0[0] - enc_repr0))
    e1 = np.mean(np.square(enc_test1[0] - enc_repr1))
    e2 = np.mean(np.square(enc_test2[0] - enc_repr2))
    e3 = np.mean(np.square(enc_test3[0] - enc_repr3))
    e4 = np.mean(np.square(enc_test4[0] - enc_repr4))
    e5 = np.mean(np.square(enc_test5[0] - enc_repr5))
    e6 = np.mean(np.square(enc_test6[0] - enc_repr6))
    e_dist = np.array([e0, e1, e2, e3, e4, e5, e6])
    min_dist = np.argsort(e_dist)[0:2]
    if yt == min_dist[0]: c1 += 1
    elif yt == min_dist[1]: c2 +=1
print "c1 = ", c1
print "c2 = ", c2
print "Total = ", c1+c2
print "Num_egs = ", len(X_test)
print "Acc = ", round((c1)/len(X_test), 2), " -- ", round((c2)/len(X_test), 2), " -- ", round((c1+c2)/len(X_test), 2)
#===============================================