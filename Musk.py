import pandas as pd
import numpy as np

# Importing data
data = pd.read_csv('musk_csv.csv')
data = data.drop(columns = ['ID'])

# Data Preprocessing
X = data.iloc[:, 2:-1].values
y = data.iloc[:, -1].values


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1
from tensorflow.keras.callbacks import TensorBoard
import time

# For tensorboard
NAME = 'musk-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs\{}'.format(NAME))


# ANN initialization
clf = Sequential()

# Addition of input and hidden layers
clf.add(Dense(activation="relu", input_dim=166, units=84, kernel_initializer="uniform", activity_regularizer=l1(0.0001)))
clf.add(Dropout(rate=0.20))

clf.add(Dense(activation="relu", units=84, kernel_initializer="uniform"))
clf.add(Dropout(rate=0.15))


# Output layer
clf.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN:
clf.fit(X_train, y_train, batch_size = 10, epochs = 10, validation_split = 0.2, callbacks = [tensorboard])



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Predicting probabilities for test set:
ycap_pred = clf.predict(X_test)
ycap_classes = clf.predict_classes(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, ycap_classes)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, ycap_classes)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, ycap_classes)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, ycap_classes)
print('F1 score: %f' % f1)
