import csv
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def fetchData(path):
    labels = []
    data = []
    f = open(path)
    csv_f = csv.reader(f)
    for row in csv_f:
        labels.append(row[0])
        data.append(row[1:])
    f.close()
    return np.array(data), np.array(labels)

def runForest(X_train, X_test, Y_train, Y_test):
    forest = RandomForestClassifier(n_estimators=90, random_state=1)
    forest = forest.fit(X_train, Y_train)
    return forest

X_test, Y_test = fetchData('data/test.csv')
X_train, Y_train = fetchData('data/train.csv')

forest = runForest(X_train, X_test, Y_train, Y_test)
score = forest.score(X_test, Y_test)
print 'Random Forest score: ', score