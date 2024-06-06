"""
Accesses the data, then creates the model and evaluates it where the results are tracked in pytest
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    roc_curve,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature

from end_to_end_utils import collect_from_database, drop_distinct

# Seed
np.random.seed(1889)


dataset_from_database = collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")

dataset_from_database = drop_distinct(dataset_from_database)

total = dataset_from_database.isnull().sum()
percent = (
    dataset_from_database.isnull().sum() / dataset_from_database.isnull().count() * 100
)
missing_df = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
types = []
for col in dataset_from_database.columns:
    dtype = str(dataset_from_database[col].dtype)
    types.append(dtype)
missing_df["Types"] = types
dataset_from_database_no_missing_values = pd.DataFrame()
dataset_from_database_no_missing_values = dataset_from_database.drop(
    columns=["family_history_3", "employment_type"]
)
dataset_from_database.drop(
    columns=["family_history_3", "employment_type"], inplace=True
)

non_numerical = [
    "gender",
    "marital_status",
    "occupation",
    "location",
    "prev_claim_rejected",
    "known_health_conditions",
    "uk_residence",
    "family_history_1",
    "family_history_2",
    "family_history_4",
    "family_history_5",
    "product_var_1",
    "product_var_2",
    "product_var_3",
    "health_status",
    "driving_record",
    "previous_claim_rate",
    "education_level",
    "income level",
    "n_dependents",
]

for column in non_numerical:
    dataset_from_database[column] = dataset_from_database[column].astype("category")


# Separate the Dataframe into labels and features
X, y = (
    dataset_from_database.drop("claim_status", axis=1),
    dataset_from_database[["claim_status"]],
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1889, stratify=y
)

# Build the evaluation set & metric list
eval_set = [(X_train, y_train)]
eval_metrics = ["auc", "rmse", "logloss"]

model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric=eval_metrics, enable_categorical=True
)

model.fit(X_test, y_test, eval_set=eval_set, verbose=10)


train_class_preds = model.predict(X_train)
test_class_preds = model.predict(X_test)
train_prob_preds = model.predict_proba(X_train)[:, 1]
test_prob_preds = model.predict_proba(X_test)[:, 1]

y = np.array(y_train)
y = y.astype(int)
yhat = np.array(train_class_preds)
yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
training_data_kappa_score = round(cohen_kappa_score(yhat, y, weights="quadratic"), 2)
print(f"The Cohen Kappa score on the training data is: {training_data_kappa_score}")

y = np.array(y_test)
y = y.astype(int)
yhat = np.array(test_class_preds)
yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
test_data_kappa_score = round(cohen_kappa_score(yhat, y, weights="quadratic"), 2)
print(f"The Cohen Kappa score on the test data is: {test_data_kappa_score}")

print()
print("The accuracy on train dataset is: ", accuracy_score(y_train, train_class_preds))
print("The accuracy on test dataset is: ", accuracy_score(y_test, test_class_preds))

print()
print("Train confusion matrix: ", confusion_matrix(y_train, train_class_preds))

print()
print("Test confusion matrix: ", confusion_matrix(y_test, test_class_preds))

print()
print("ROC on train data: ", roc_auc_score(y_train, train_prob_preds))
print("ROC on test data: ", roc_auc_score(y_test, test_prob_preds))

print()
fpr, tpr, _ = roc_curve(y_test, test_prob_preds)
random_fpr, random_tpr, _ = roc_curve(y_test, [0 for _ in range(len(y_test))])
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(fpr, tpr, marker=".", label="XGBoost")
plt.plot(random_fpr, random_tpr, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Curve")
print("Train log loss: ", log_loss(y_train, train_prob_preds))
print("Test log loss: ", log_loss(y_test, test_prob_preds))

print()
print("F1 score is: ", f1_score(y_test, test_class_preds))
print("Precision is: ", precision_score(y_test, test_class_preds))
print("Recall is: ", recall_score(y_test, test_class_preds))

parameter_gridSearch = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=eval_metrics,
        early_stopping_rounds=15,
        enable_categorical=True,
    ),
    param_distributions={
        "n_estimators": stats.randint(50, 500),
        "learning_rate": stats.uniform(0.01, 0.75),
        "subsample": stats.uniform(0.25, 0.75),
        "max_depth": stats.randint(1, 5),
        "colsample_bytree": stats.uniform(0.1, 0.75),
        "min_child_weight": [1, 3, 5, 7, 9],
    },
    cv=5,
    n_iter=100,
    verbose=False,
    scoring="roc_auc",
)

parameter_gridSearch.fit(X_train, y_train, eval_set=eval_set, verbose=False)

print("Best parameters are: ", parameter_gridSearch.best_params_)

model3 = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric=eval_metrics,
    early_stopping_rounds=15,
    enable_categorical=True,
    **parameter_gridSearch.best_params_,  # Not sure what this does, from StackOverflow
)

model3.fit(X_train, y_train, eval_set=eval_set, verbose=False)

train_class_preds2 = model3.predict(X_train)
test_class_preds2 = model3.predict(X_test)
train_prob_preds2 = model3.predict_proba(X_train)[:, 1]
test_prob_preds2 = model3.predict_proba(X_test)[:, 1]

y = np.array(y_train)
y = y.astype(int)
yhat = np.array(train_class_preds2)
yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
kappa2 = round(cohen_kappa_score(yhat, y, weights="quadratic"), 2)
print(f"The Cohen Kappa score on the training data is: {kappa2}")

y = np.array(y_test)
y = y.astype(int)
yhat = np.array(test_class_preds)
yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
kappa2 = round(cohen_kappa_score(yhat, y, weights="quadratic"), 2)
print(f"The Cohen Kappa score on the test data is: {kappa2}")

print()

training_accuracy = accuracy_score(y_train, train_class_preds2)
print("The accuracy on train dataset is: ", training_accuracy)

test_accuracy = accuracy_score(y_test, test_class_preds2)
print("The accuracy on test dataset is: ", test_accuracy)

print()
print("Train confusion matrix: ", confusion_matrix(y_train, train_class_preds2))

print()
print("Test confusion matrix: ", confusion_matrix(y_test, test_class_preds2))

print()
print("ROC on train data: ", roc_auc_score(y_train, train_prob_preds2))
print("ROC on test data: ", roc_auc_score(y_test, test_prob_preds2))

print()

print("Train log loss: ", log_loss(y_train, train_prob_preds2))
print("Test log loss: ", log_loss(y_test, test_prob_preds2))

print()

f1_score = f1_score(y_test, test_class_preds2)
print("F1 score is: ", f1_score)

precision_score = precision_score(y_test, test_class_preds2)

print("Precision is: ", precision_score)


recall_score = recall_score(y_test, test_class_preds2)
print("Recall is: ", recall_score)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(parameter_gridSearch.best_params_)

    # Log the loss metric
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("training_accuracy", training_accuracy)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("precision_score", precision_score)
    mlflow.log_metric("recall_score", recall_score)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "With Stratify = y and max_depth = 5")

    # Infer the model signature
    signature = infer_signature(X_train, parameter_gridSearch.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=parameter_gridSearch,
        artifact_path="credit_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
