"""
main.py
-------
Entry point for the whole project. Run this from VS Code (or `python main.py`
from the project root) to execute the complete pipeline from scratch:

    1. Load the dataset                     (src/data_loader.py)
    2. Basic checks                          (src/data_loader.py)
    3. Exploratory Data Analysis             (src/eda.py)
    4. Train/test split + scaling            (src/train_models.py)
    5. Train all 6 classifiers               (src/train_models.py)
    6. Compare models + pick the best one    (src/train_models.py)
    7. Feature importance                    (src/feature_importance.py)
    8. Save the best model for the Flask app (models/model_bundle.pkl)

After this finishes, start the web app with:
    cd frontend
    python app.py
"""

from src.data_loader import basic_checks, load_data
from src.eda import run_full_eda
from src.feature_importance import plot_feature_importance
from src.train_models import (compare_models, prepare_data, save_best_model,
                                train_all_models)


def main():
    print("=" * 70)
    print("PRCP-1009 : CELLPHONE PRICE PREDICTION — FULL PIPELINE")
    print("=" * 70)

    # 1-2. Load data + basic checks
    df = load_data()
    basic_checks(df)

    # 3. EDA (charts saved to outputs/eda/)
    run_full_eda(df)

    # 4. Prepare data for modelling
    (X, y, X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = prepare_data(df)

    # 5. Train all six classifiers
    results, trained_models = train_all_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    # 6. Compare models and select the best one
    results_df = compare_models(results)
    best_model_name = save_best_model(results_df, trained_models, scaler, X.columns)

    # 7. Feature importance (always computed from Random Forest,
    #    regardless of which model ends up "best", since it's the
    #    most reliable importance signal available)
    plot_feature_importance(trained_models["Random Forest"], X.columns)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE. Best model: {best_model_name}")
    print("All charts saved under 'outputs/'. Model saved under 'models/'.")
    print("Next: cd frontend && python app.py   (then open http://127.0.0.1:5000)")
    print("=" * 70)


if __name__ == "__main__":
    main()
