from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
import json
import os


def NCA():
    root_dir = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'Tennis_AI', 'Tennis_AI_Complete')

    with open(os.path.join(root_dir, "Supervised_Learning", "Angle_Data.json"), "r") as file:
        angles = json.load(file)

    x = []
    y = []
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            x.append(angles[i][j])
            y.append(i)

    nca = NeighborhoodComponentsAnalysis(random_state=50)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(x, y)

    return nca_pipe
