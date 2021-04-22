import os
import pickle

import numpy as np
from rubicon import Rubicon
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def _load_penguins():
    print("🐧 loading penguins...")
    from palmerpenguins import load_penguins

    X, y = load_penguins(return_X_y=True)
    print(X.head())

    return X, y


def _log_cv_results_to_rubicon(fit_grid_search):
    rubicon = Rubicon(
        persistence="filesystem",
        root_dir=f"{os.getcwd()}/rubicon-root",
        auto_git_enabled=True,
    )
    project = rubicon.get_or_create_project("Classifying Penguins")

    best_estimator = fit_grid_search.best_estimator_
    best_estimator_bytes = pickle.dumps(best_estimator)

    project.log_artifact(
        data_bytes=best_estimator_bytes,
        name="KNeighborsClassifier best estimator",
    )

    cv_results = fit_grid_search.cv_results_

    for i, score in enumerate(cv_results.get("mean_test_score")):
        experiment = project.log_experiment(model_name="KNeighborsClassifier")

        parameters = cv_results.get("params")[i]
        for name, value in parameters.items():
            experiment.log_parameter(name=name, value=value)

        experiment.log_metric(name="accuracy", value=score)


X, y = _load_penguins()

imputer = SimpleImputer(missing_values=np.nan)
classifier = KNeighborsClassifier()

pipeline = Pipeline([("imp", imputer), ("clf", classifier)])
param_grid = {
    "clf__n_neighbors": (5, 10, 15, 20),
    "clf__weights": ("uniform", "distance"),
    "clf__algorithm": ("ball_tree", "kd_tree", "brute"),
}

print(f"fitting `KNeighborsClassifier`s...")
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1)
grid_search.fit(X, y)

print(f"results:\n{grid_search.cv_results_}")

print("logging results to Rubicon...")
_log_cv_results_to_rubicon(grid_search)
