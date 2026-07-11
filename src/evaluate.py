import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from preprocess import preprocess, load_config

def load_best_model():
    model = joblib.load("models/best_model.pkl")
    print("✅ Best model loaded!")
    return model

def evaluate_model(model, X_test, y_test):
    print("\n📊 Evaluating model on test data...\n")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }
    print("=" * 40)
    print("📈 MODEL PERFORMANCE ON TEST DATA")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"   {metric.upper():<12}: {value:.4f}")
    print("=" * 40)
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    os.makedirs("notebooks", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good Customer", "Bad Customer"],
        yticklabels=["Good Customer", "Bad Customer"]
    )
    plt.title("Confusion Matrix — Credit Scoring Model", fontsize=14)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("notebooks/confusion_matrix.png")
    print("✅ Confusion matrix saved!")
    plt.show()

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — Credit Scoring Model", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/roc_curve.png")
    print("✅ ROC curve saved!")
    plt.show()

def explain_with_shap(model, X_test, feature_names):
    print("\n🔍 Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, 1],
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Feature Importance — Credit Scoring")
    plt.tight_layout()
    plt.savefig("notebooks/shap_summary.png")
    print("✅ SHAP summary saved!")
    plt.show()

def run_evaluation():
    config = load_config()
    X_train, X_test, y_train, y_test = preprocess(config)
    model = load_best_model()
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    feature_names = [
        "Age", "Sex", "Job", "Housing",
        "Saving accounts", "Checking account",
        "Credit amount", "Duration", "Purpose",
        "credit_per_month", "age_duration_ratio"
    ]
    explain_with_shap(model, X_test, feature_names)
    return metrics

if __name__ == "__main__":
    run_evaluation()
    print("\n🎉 Evaluation complete! Check notebooks/ for plots.")