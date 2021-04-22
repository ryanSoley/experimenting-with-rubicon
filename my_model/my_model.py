import os
import pickle

import numpy as np
from rubicon import Rubicon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
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
        name="RandomForrestClassifier best estimator",
    )

    cv_results = fit_grid_search.cv_results_

    for i, score in enumerate(cv_results.get("mean_test_score")):
        experiment = project.log_experiment(model_name="RandomForestClassifier")

        parameters = cv_results.get("params")[i]
        for name, value in parameters.items():
            experiment.log_parameter(name=name, value=value)

        experiment.log_metric(name="accuracy", value=score)


X, y = _load_penguins()

imputer = SimpleImputer(missing_values=np.nan)
classifier = RandomForestClassifier()

pipeline = Pipeline([("imp", imputer), ("clf", classifier)])
param_grid = {
    "clf__max_depth": (5, 10, 15, 20),
    "clf__n_estimators": (2, 4, 6, 8, 10),
}

print(f"fitting `RandomForestClassifier`s...")
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1)
grid_search.fit(X, y)

print(f"results:\n{grid_search.cv_results_}")

print("logging results to Rubicon...")
_log_cv_results_to_rubicon(grid_search)
