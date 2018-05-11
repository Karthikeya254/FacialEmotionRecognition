from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import numpy as np
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

def top2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

alexnet = load_model('Final_models/Alexnet.h5', custom_objects={'top2_acc': top2_acc})
resnet = load_model('Final_models/Affectnet_resnet-FullData-run1.h5', custom_objects={'top2_acc': top2_acc})
vggnet = load_model('Final_models/VGGnet-FullData-batch-Run_qsub.h5', custom_objects={'top2_acc': top2_acc})
vggnet_dark = load_model('Final_models/VGGnet-FullData-batch-Run_dark_1.h5', custom_objects={'top2_acc': top2_acc})
alexnet_dark = load_model('Final_models/Darknet-AlexnetGray-Run1.h5', custom_objects={'top2_acc': top2_acc})

alex_pred_probs = alexnet.predict(X_test)
res_pred_probs = resnet.predict(X_test)
vgg_pred_probs = vggnet.predict(X_test)
alexd_pred_probs = alexnet_dark.predict(X_test)
vggd_pred_probs = vggnet_dark.predict(X_test)

alexnet.evaluate(X_test, Y_test, verbose=0)
vggnet.evaluate(X_test, Y_test, verbose=0)
resnet.evaluate(X_test, Y_test, verbose=0)
alexnet_dark.evaluate(X_test, Y_test, verbose=0)
vggnet_dark.evaluate(X_test, Y_test, verbose=0)

test_classes = np.argmax(Y_test, axis=1)
alex_pred_classes = np.argmax(alex_pred_probs, axis=1)
res_pred_classes = np.argmax(res_pred_probs, axis=1)
vgg_pred_classes = np.argmax(vgg_pred_probs, axis=1)
alexd_pred_classes = np.argmax(alexd_pred_probs, axis=1)
vggd_pred_classes = np.argmax(vggd_pred_probs, axis=1)

# Top-1 Accuracies of all model predictions
print("Alexnet:",accuracy_score(alex_pred_classes, test_classes))
print("Resnet:",accuracy_score(res_pred_classes, test_classes))
print("Vggnet:",accuracy_score(vgg_pred_classes, test_classes))
print("Alexdark:",accuracy_score(alexd_pred_classes, test_classes))
print("Vggdark:",accuracy_score(vggd_pred_classes, test_classes))

# Ensemble of all large networks (VGG, Alexnet, Resnet)
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
print("Ensemble Large:",accuracy_score(ensemble1_pred_classes, test_classes))
#Top-2 accuracy of ensembled network
print("Top2 Ensemble Acc:", pred_count/len(test_classes))

# Ensemble of smaller networks
ensembled1_pred_classes = []
ensembled2_pred_classes = []
for i in range(len(vggd_pred_classes)):
    md = mode([vggd_pred_classes[i], alexd_pred_classes[i], res_pred_classes[i]])
    if md.count[0] > 1:
        ensembled1_pred_classes.append(md.mode[0])
        if len(md.mode) > 1:
            ensembled2_pred_classes.append(md.mode[1])
        else:
            ensembled2_pred_classes.append(md.mode[0])
    else:
        ensembled1_pred_classes.append(vggd_pred_classes[i])
        ensembled2_pred_classes.append(alexd_pred_classes[i])

pred_count = 0
for i in range(len(test_classes)):
    if test_classes[i] in [ensembled1_pred_classes[i], ensembled2_pred_classes[i]]:
        pred_count += 1

#Top-1 accuracy of ensembled network
print("Ensemble Small:",accuracy_score(ensembled1_pred_classes, test_classes))
#Top-2 accuracy of ensembled network
print("Top2 Ensemble Acc Dark:", pred_count/len(test_classes))