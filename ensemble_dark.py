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
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from utils import getData, top2_acc, prepareCallbacks, printMetrics

sess=tf.Session()
set_session(sess)

path = "Affectnet_data/"
X_train, X_test, Y_train, Y_test = getData(path+"Affectnet-96X-Full.npy", path+"Affectnet-Y-Full.npy", 0.2)

alexnet_dark = load_model('Final_models/VGG_Alex_dark.h5', custom_objects={'top2_acc': top2_acc})
resnet_dark = load_model('Final_models/VGG_Res_dark.h5', custom_objects={'top2_acc': top2_acc})
vggnet_dark = load_model('Final_models/VGG_VGG_dark.h5', custom_objects={'top2_acc': top2_acc})

alex_pred_probs = alexnet_dark.predict(X_test)
res_pred_probs = resnet_dark.predict(X_test)
vgg_pred_probs = vggnet_dark.predict(X_test)

alexnet_dark.evaluate(X_test, Y_test, verbose=0)
resnet_dark.evaluate(X_test, Y_test, verbose=0)
vggnet_dark.evaluate(X_test, Y_test, verbose=0)

test_classes = np.argmax(Y_test, axis=1)
alex_pred_classes = np.argmax(alex_pred_probs, axis=1)
res_pred_classes = np.argmax(res_pred_probs, axis=1)
vgg_pred_classes = np.argmax(vgg_pred_probs, axis=1)

# Top-1 Accuracies of all model predictions
print("Alexnet Dark:",accuracy_score(alex_pred_classes, test_classes))
print("Resnet Dark:",accuracy_score(res_pred_classes, test_classes))
print("Vggnet Dark:",accuracy_score(vgg_pred_classes, test_classes))

# Ensemble of all dark networks (VGG, Alexnet, Resnet)
ensemble1_pred_classes = []
ensemble2_pred_classes = []
for i in range(len(vgg_pred_classes)):
    md = mode([vgg_pred_classes[i], alex_pred_classes[i], res_pred_classes[i]])
    if md.count[0] > 1:
        ensemble1_pred_classes.append(md.mode[0])
        if len(md.mode) > 1:
            ensemble2_pred_classes.append(md.mode[1])
        else:
            ensemble2_pred_classes.append(md.mode[0])
    else:
        ensemble1_pred_classes.append(vgg_pred_classes[i])
        ensemble2_pred_classes.append(alex_pred_classes[i])
pred_count = 0
for i in range(len(test_classes)):
    if test_classes[i] in [ensemble1_pred_classes[i], ensemble2_pred_classes[i]]:
        pred_count += 1
#Top-1 accuracy of ensembled network
print("Ensemble Dark:",accuracy_score(ensemble1_pred_classes, test_classes))
#Top-2 accuracy of ensembled network
print("Top2 Ensemble Acc Dark:", pred_count/len(test_classes))