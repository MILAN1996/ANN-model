# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense    #initialize random variables by dense
from keras.layers import Dropout 
# Initialising the ANN
classifier = Sequential()

#Adding the input layer and 1st hidden layer

#output_dim = no of nodes in hidden layer
#init = uniform initial weights
#input_dim = input variables
classifier.add(Dense(output_dim = 100,init = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dropout(p=0.1))
#Adding 2nd hidden layer
classifier.add(Dense(output_dim = 100,init = 'uniform',activation = 'relu'))
classifier.add(Dropout(p=0.1))
#Adding output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics= ['accuracy'])

#batches and epochs
classifier.fit(x_train,y_train, batch_size = 10,epochs =5)

#making prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#single customer test
new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,2,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 5)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

#dropout

#parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
     classifier = Sequential()
     classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
     classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[10],
                  'epochs':[5,10,15],
                  'optimizer':['adam']}
 
grid_search = GridSearchCV(estimator = classifier,param_grid= parameters,scoring = 'accuracy',cv =10,n_jobs=1)
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_accuracy)
print(mean)
print(best_parameters)
