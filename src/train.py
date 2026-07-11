import numpy as np
import joblib
import yaml
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from preprocess import preprocess, load_config

def get_models():
    """
    🧠 Define all 3 models with their hyperparameters.
    Hyperparameters = settings we choose BEFORE training.
    Like choosing how many trees in a forest before planting it.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,      # Max attempts to find best line
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,   # 100 trees in the forest
            max_depth=10,       # How deep each tree can grow
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,   # 100 boosting rounds
            max_depth=6,        # Tree depth
            learning_rate=0.1,  # How fast it learns (too fast = misses patterns)
            random_state=42,
            eval_metric="logloss",
            verbosity=0         # Suppress XGBoost logs
        )
    }
    return models

def train_models(X_train, y_train, config):
    """
    🧠 Train each model and evaluate using Cross Validation.
    
    Cross Validation = instead of training once, we train 5 times
    on different splits of data and average the score.
    Why? So we know the model actually learned, not just got lucky
    on one particular split.
    """
    models = get_models()
    trained_models = {}
    cv_folds = config["model"]["cv_folds"]  # 5 from config.yaml

    print("\n🚀 Starting model training...\n")

    for name, model in models.items():
        print(f"Training {name}...")

        # Cross validation score
        # 🧠 cv=5 means split data into 5 parts, train on 4, test on 1, repeat 5 times
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring="roc_auc"   # ROC-AUC is best metric for imbalanced classification
        )

        print(f"   CV ROC-AUC scores: {cv_scores.round(3)}")
        print(f"   Mean: {cv_scores.mean():.3f} | Std: {cv_scores.std():.3f}")

        # Train on full training data
        model.fit(X_train, y_train)

        # Save model to disk
        # 🧠 joblib.dump = pickle the model into a file so we can load it later
        model_path = os.path.join(config["paths"]["model_save_path"], f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"   ✅ Saved to {model_path}\n")

        trained_models[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }

    # Find best model based on CV score
    best_name = max(trained_models, key=lambda x: trained_models[x]["cv_mean"])
    best_model = trained_models[best_name]["model"]

    print(f"🏆 Best model: {best_name} with CV ROC-AUC: {trained_models[best_name]['cv_mean']:.3f}")

    # Save best model separately
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"✅ Best model saved to models/best_model.pkl")

    return trained_models, best_name

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = preprocess(config)
    trained_models, best_name = train_models(X_train, y_train, config)
    print("\n🎉 Training complete!")