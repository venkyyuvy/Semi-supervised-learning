from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
iris = datasets.load_iris()
rng = np.random.RandomState(42)
n = len(iris.data)
scores = []
unlabel_ratios = np.arange(0.1, 0.95, 0.1)
for unlabel in unlabel_ratios:
    label_prop_model = LabelPropagation()
    random_unlabeled_points = rng.choice(n, int(n*unlabel), False)
    labels = np.copy(iris.target)
    labels[random_unlabeled_points] = -1
    label_prop_model.fit(iris.data, labels)
    y_pred = label_prop_model.predict(
        np.copy(iris.data)[random_unlabeled_points])
    scores.append(metrics.f1_score(y_pred,
                                   np.copy(iris.target)[
                                       random_unlabeled_points],
                                   average="micro"))

fig, ax = plt.subplots()
p = ax.plot(unlabel_ratios, scores)
ax.set_ylim([0.0, 1.1])
ax.set_ylabel('accuracy on unlabelled data')
ax.set_xlabel('percentage of unlabelled data')

fig.savefig('label_prop.png')
