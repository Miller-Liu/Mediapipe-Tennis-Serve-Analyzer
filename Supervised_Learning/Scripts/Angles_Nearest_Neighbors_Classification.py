from sklearn import neighbors
import json
import os


def NNC():
    root_dir = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'Tennis_AI', 'Tennis_AI_Complete')

    with open(os.path.join(root_dir, "Supervised_Learning", "Angle_Data.json"), "r") as file:
        angles = json.load(file)

    x = []
    y = []
    n_neighbors = 15
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            x.append(angles[i][j])
            y.append(i)

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
    clf.fit(x, y)

    return clf
