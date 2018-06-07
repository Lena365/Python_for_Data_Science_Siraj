from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female',
     'female', 'female', 'male', 'male']

DT = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
NB = GaussianNB()
# 2
KN = KNeighborsClassifier(n_neighbors = 3)
# 3
RF = RandomForestClassifier()

# Train the modules on our data
DT = DT.fit(X, Y)
NB = NB.fit(X, Y)
KN = KN.fit(X, Y)
RF = RF.fit(X, Y)

pred_DT = DT.predict(X)
pred_NB = NB.predict(X)
pred_KN = KN.predict(X)
pred_RF = RF.predict(X)

# Compare the modules
A = [accuracy_score(Y, pred_DT), accuracy_score(Y, pred_NB), 
     accuracy_score(Y, pred_KN), accuracy_score(Y, pred_RF)]
A = [a * 100 for a in A]
A = [round(a, 3) for a in A]

A_name = ['DecisionTreeClassifier', 'GaussianNB', 
          'KNeighborsClassifier','RandomForestClassifier']
for i in range(4):
    if A[i] == max(A):
       print(A_name[i] + ': {}'.format(A[i]) + '*')
    else:
       print(A_name[i] + ': {}'.format(A[i]))

# The best classifier from DT, NB, KN, RF
index = np.argmax(A)
classifiers = {0: 'DecisionTreeClassifier', 1: 'GaussianNB', 
               2: 'KNeighborsClassifier', 3: 'RandomForestClassifier'}
print('The best gender classifier is {}'.format(classifiers[index]))

# Print result for sample
sample = [190, 70, 43]
prediction = DT.predict([sample])
print('The best prediction of sample ' + str(sample) + ' is ')
print(prediction)
