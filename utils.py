import numpy as np
from sklearn.cross_validation import train_test_split
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

def getData(ipfile, opfile, test_split):
	X = np.load(ipfile)
	Y = np.load(opfile)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split, random_state=42)
	Y_train = np.eye(np.unique(Y_train).shape[0])[Y_train]
	Y_test = np.eye(np.unique(Y_test).shape[0])[Y_test]
	X_train = X_train.astype('float32')
	X_train /= 255
	X_test = X_test.astype('float32')
	X_test /= 255
	return(X_train, X_test, Y_train, Y_test)

def top2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def prepareCallbacks(model_path, logger_file):
	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	early_stopper = EarlyStopping(monitor='val_acc',min_delta=0.001, patience=20)
	csv_logger = CSVLogger(logger_file)
	model_checkpt= ModelCheckpoint(model_path,monitor='val_acc',save_best_only=True, mode='max',verbose=0)
	return(lr_reducer, early_stopper, csv_logger, model_checkpt)

def printMetrics(scores):
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	print('Test Top-2 accuracy:', scores[2])
