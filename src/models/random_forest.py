from sklearn.ensemble import RandomForestClassifier


def create_random_forest(
    n_estimators=10,
    random_state=42
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1
    )