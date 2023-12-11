import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler


def distance(x, y):
    return np.sum(np.abs(x - y))


def taxicab_sample(n, r):
    sample = []

    for _ in range(n):
        spread = r - np.sum([np.abs(x) for x in sample])
        sample.append(spread * (2 * np.random.rand() - 1))

    return np.random.permutation(sample)


class CCR:
    def __init__(self, energy=0.25, scaling=0.0, n=None, sampling_strategy=None):
        self.energy = energy
        self.scaling = scaling
        self.n = n
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
            X = X.values
        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]

        assert len(classes) == len(set(sizes)) == 2

        minority_class = classes[np.argmin(sizes)]
        majority_class = classes[np.argmax(sizes)]
        minority = X[y == minority_class]
        majority = X[y == majority_class]

        if self.n is None:
            n = len(majority) - len(minority)
        else:
            n = self.n

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = np.zeros((len(minority), len(majority)))

        for i in range(len(minority)):
            for j in range(len(majority)):
                distances[i][j] = distance(minority[i], majority[j])

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy
            r = 0.0
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            while True:
                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change

                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                if (
                    distances[i, sorted_distances[current_majority]]
                    >= r + radius_change
                ):
                    r += radius_change

                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[
                            i, sorted_distances[current_majority - 1]
                        ]

                    radius_change = (
                        distances[i, sorted_distances[current_majority]] - last_distance
                    )
                    r += radius_change
                    remaining_energy -= radius_change * (current_majority + 1.0)
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    majority_point += (
                        1e-6 * np.random.rand(len(majority_point)) + 1e-6
                    ) * np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority += translations

        appended = []

        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(
                np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n)
            )
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        return np.concatenate([majority, minority, appended]), np.concatenate(
            [
                np.tile([majority_class], len(majority)),
                np.tile([minority_class], len(minority) + len(appended)),
            ]
        )


class CCRSelection:
    def __init__(
        self,
        classifier,
        measure,
        n_splits=5,
        energies=(0.25,),
        scaling_factors=(0.0,),
        n=None,
    ):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.energies = energies
        self.scaling_factors = scaling_factors
        self.n = n
        self.selected_energy = None
        self.selected_scaling = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_resample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for energy in self.energies:
            for scaling in self.scaling_factors:
                scores = []

                for train_idx, test_idx in self.skf.split(X, y):
                    X_train, y_train = CCR(
                        energy=energy, scaling=scaling, n=self.n
                    ).fit_resample(X[train_idx], y[train_idx])

                    classifier = self.classifier.fit(X_train, y_train)
                    predictions = classifier.predict(X[test_idx])
                    scores.append(self.measure(y[test_idx], predictions))

                score = np.mean(scores)

                if score > best_score:
                    self.selected_energy = energy
                    self.selected_scaling = scaling

                    best_score = score

        return CCR(
            energy=self.selected_energy, scaling=self.selected_scaling, n=self.n
        ).fit_resample(X, y)


def _rbf(d, eps):
    return np.exp(-((d * eps) ** 2))


def _distance(x, y):
    return np.sum(np.abs(x - y))


def _pairwise_distances(X):
    D = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue

            d = _distance(X[i], X[j])

            D[i][j] = d
            D[j][i] = d

    return D


def _score(point, X, y, minority_class, epsilon):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbf = _rbf(_distance(point, X[i]), epsilon)

        if y[i] == minority_class:
            mutual_density_score -= rbf
        else:
            mutual_density_score += rbf

    return mutual_density_score


class RBO:
    def __init__(
        self,
        gamma=0.05,
        n_steps=500,
        step_size=0.001,
        stop_probability=0.02,
        criterion="balance",
        minority_class=None,
        n=None,
        sampling_strategy=None,
    ):
        assert criterion in ["balance", "minimize", "maximize"]
        assert 0.0 <= stop_probability <= 1.0

        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = minority_class
        self.n = n
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
            y = y.copy().values
        epsilon = 1.0 / self.gamma
        classes = np.unique(y)

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class]

        if self.n is None:
            n = sum(y != minority_class) - sum(y == minority_class)
        else:
            n = self.n

        if n == 0:
            return X, y

        minority_scores = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            minority_scores.append(
                _score(minority_point, X, y, minority_class, epsilon)
            )

        appended = []

        while len(appended) < n:
            idx = np.random.choice(range(len(minority_points)))
            point = minority_points[idx].copy()
            score = minority_scores[idx]

            for i in range(self.n_steps):
                if (
                    self.stop_probability is not None
                    and self.stop_probability > np.random.rand()
                ):
                    break

                translation = np.zeros(len(point))
                sign = np.random.choice([-1, 1])
                translation[np.random.choice(range(len(point)))] = sign * self.step_size
                translated_point = point + translation
                translated_score = _score(
                    translated_point, X, y, minority_class, epsilon
                )

                if (
                    (
                        self.criterion == "balance"
                        and np.abs(translated_score) < np.abs(score)
                    )
                    or (self.criterion == "minimize" and translated_score < score)
                    or (self.criterion == "maximize" and translated_score > score)
                ):
                    point = translated_point
                    score = translated_score

            appended.append(point)

        return np.concatenate([X, appended]), np.concatenate(
            [y, minority_class * np.ones(len(appended))]
        )


class RBOSelection:
    def __init__(
        self,
        classifier,
        measure,
        n_splits=5,
        gammas=(0.05,),
        n_steps=500,
        step_size=0.001,
        stop_probability=0.02,
        criterion="balance",
        minority_class=None,
        n=None,
    ):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.gammas = gammas
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = minority_class
        self.n = n
        self.selected_gamma = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_resample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for gamma in self.gammas:
            scores = []

            for train_idx, test_idx in self.skf.split(X, y):
                X_train, y_train = RBO(
                    gamma=gamma,
                    n_steps=self.n_steps,
                    step_size=self.step_size,
                    stop_probability=self.stop_probability,
                    criterion=self.criterion,
                    minority_class=self.minority_class,
                    n=self.n,
                ).fit_resample(X[train_idx], y[train_idx])

                classifier = self.classifier.fit(X_train, y_train)
                predictions = classifier.predict(X[test_idx])
                scores.append(self.measure(y[test_idx], predictions))

            score = np.mean(scores)

            if score > best_score:
                self.selected_gamma = gamma

                best_score = score

        return RBO(
            gamma=self.selected_gamma,
            n_steps=self.n_steps,
            step_size=self.step_size,
            stop_probability=self.stop_probability,
            criterion=self.criterion,
            minority_class=self.minority_class,
            n=self.n,
        ).fit_resample(X, y)


def rbf(d, eps):
    return np.exp(-((d * eps) ** 2))


def pairwise_distances(X):
    D = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue

            d = distance(X[i], X[j])

            D[i][j] = d
            D[j][i] = d

    return D


def score(point, X, epsilon):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbfRes = rbf(distance(point, X[i, :]), epsilon)
        mutual_density_score += rbfRes

    return mutual_density_score


def scoreAll(points, X, epsilon):
    cur_mutual_density_score = 0.0
    mutual_density_scores = []

    for j in range(len(points)):
        point = points[j, :]

        for i in range(len(X)):
            rbfRes = rbf(distance(point, X[i, :]), epsilon)
            cur_mutual_density_score += rbfRes

        mutual_density_scores = np.append(
            mutual_density_scores, cur_mutual_density_score
        )
        cur_mutual_density_score = 0.0

    return mutual_density_scores


class SwimRBF:
    def __init__(
        self, minCls=None, epsilon=None, steps=5, tau=0.25, sampling_strategy=None
    ):
        self.epsilon = epsilon
        self.steps = steps
        self.tau = tau
        self.minCls = minCls
        self.scaler = StandardScaler()
        self.sampling_strategy = sampling_strategy

    def extremeRBOSample(self, data, labels, numSamples=None):
        if isinstance(data, pd.DataFrame):
            cols = data.columns.tolist()
            data = data.values
        classes = np.unique(labels)
        sizes = [sum(labels == c) for c in classes]

        assert len(classes) == len(set(sizes)) == 2

        if self.minCls is None:
            self.minCls = np.argmin(np.bincount(labels.astype(int)))
        self.maxCls = np.argmax(np.bincount(labels.astype(int)))

        trnMajData = data[np.where(labels != self.minCls)[0], :]
        trnMinData = data[np.where(labels == self.minCls)[0], :]

        if numSamples is None:
            numSamples = np.sum(labels == self.maxCls) - np.sum(labels == self.minCls)

        if self.epsilon is None:
            self.epsilon = self.fit(trnMajData)

        synthData = np.empty([0, data.shape[1]])
        stds = self.tau * np.std(trnMinData, axis=0)

        if np.sum(labels == self.minCls) == 1:
            trnMinData = trnMinData.reshape(1, len(trnMinData))

        while synthData.shape[0] < numSamples:
            j = np.random.choice(trnMinData.shape[0], 1)[0]
            scoreCur = score(trnMinData[j, :], trnMajData, self.epsilon)
            for k in range(self.steps):
                step = trnMinData[j, :] + np.random.normal(0, stds, trnMinData.shape[1])
                stepScore = score(step, trnMajData, self.epsilon)
                if stepScore <= scoreCur:
                    synthData = np.append(
                        synthData, step.T.reshape((1, len(step))), axis=0
                    )
                    break

        sampled_data = np.concatenate([np.array(synthData), data])
        sampled_labels = np.append([self.minCls] * len(synthData), labels)

        return sampled_data, sampled_labels

    def fit_resample(self, data, labels):
        return self.extremeRBOSample(data, labels)

    def fit(self, data):
        d = pdist(data)
        return 0.5 * np.std(d) * np.mean(d)
