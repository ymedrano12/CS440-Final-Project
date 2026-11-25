from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def create_logistic_regression(
    C=1.0,
    max_iter=1000,
    random_state=42
):
    return OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state
        ),
        n_jobs=1
    )