from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC


# Loading an example dataset

iris = datasets.load_iris()
digits = datasets.load_digits()

# A dataset is a dictionary like object, holding the data
# and some metadata.

# .data member is a n-samples, n-features array.
print(digits.data)
# .target member stores the response variable.
print(digits.target)
# Each original sample is an image of shape (8,8)
print(digits.images[0])

# Learning and predicting

# Here, we are given samples of each of the 10 possible classes
# (the digits zero through nine) on which we fit an estimator
# to be able to predict the classes of new samples

# An estimator is a Python object that implements the methods
# fit(x, y) and predict(T)
# Ex: Support Vector Classification
# Here gamma is set manually. But to find good values for
# these parameters, we can use tools like grid search
# or cross validation
clf = svm.SVC(gamma=0.001, C=100.)

# The fitting, or 'learning' from the model
# Here, the training set: all the images but the last one for
# the target (prediction)
clf.fit(digits.data[:-1], digits.target[:-1])

# Predicting from the last image
clf.predict(digits.data[-1:])

# Model persistence

# Save a model in scikit-learn with using 'pickle' Python's persistence
clf = svm.SVC(gamma='scale')
X, y = iris.data, iris.target
clf.fit(X, y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

# Using joblib replacement for pickle
from joblib import dump, load
dump(clf, 'filename.joblib')
clf = load('filename.joblib')

# Conventions

# Unless otherwise specified, input will be cast to float64
# Classification targets are maintained
clf = SVC(gamma='scale')
clf.fit(iris.data, iris.target)
list(clf.predict(iris.data[:3]))
# [0, 0, 0]
clf.fit(iris.data, iris.target_names[iris.target])
list(clf.predict(iris.data[:3]))
# ['setosa', 'setosa', 'setosa']

# Refitting and updating parameters

from sklearn.datasets import load_iris

