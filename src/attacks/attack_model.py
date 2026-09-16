
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.attacks.features import extract_mia_features


def train_attack_model(
    attack_df,
    feature_cols,
    label_col="ground_truth",
    test_size=0.15,
    random_state=42,
):
    X = attack_df[feature_cols]
    y = attack_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Attack model validation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n", confusion_matrix(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred))

    return model


def infer_membership(
    synthetic_df,
    target_user_trips,
    mia_attack_model,
    feature_cols,
):
    feats = extract_mia_features(
        synthetic_df=synthetic_df,
        target_user_trips=target_user_trips,
    )

    X = pd.DataFrame([feats])[feature_cols]

    pred = mia_attack_model.predict(X)[0]
    prob = mia_attack_model.predict_proba(X)[0, 1]

    return {
        "prediction": pred,
        "membership_probability": prob,
        **feats,
    }


def run_attack_evaluation(
    synthetic_df,
    users,
    ground_truth_labels,
    mia_attack_model,
    feature_cols,
):
    rows = []

    for user_id, target_user_trips in users.items():

        result = infer_membership(
            synthetic_df=synthetic_df,
            target_user_trips=target_user_trips,
            mia_attack_model=mia_attack_model,
            feature_cols=feature_cols,
        )

        result["user_id"] = user_id
        result["ground_truth"] = ground_truth_labels[user_id]

        rows.append(result)

    results_df = pd.DataFrame(rows)

    y_true = results_df["ground_truth"]
    y_pred = results_df["prediction"]

    print("Final attack evaluation")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\n", confusion_matrix(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred))

    return results_df
