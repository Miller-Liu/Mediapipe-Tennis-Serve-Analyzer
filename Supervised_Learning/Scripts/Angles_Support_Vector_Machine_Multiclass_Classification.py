import json
from sklearn import svm
from sklearn.model_selection import train_test_split
import os


def SVM():
    root_dir = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'Tennis_AI', 'Tennis_AI_Complete')

    with open(os.path.join(root_dir, "Supervised_Learning", "Angle_Data.json"), "r") as file:
        angles = json.load(file)

    x = []
    y = []
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            x.append(angles[i][j])
            y.append(i)

    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1, random_state=50)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)

    return clf
