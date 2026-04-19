"""
Run a single experiment.

This file is a thin orchestration layer:
- Loads data via prepare.py
- Evaluates the current model (from model.py)
- Logs results to results.tsv
- Updates performance.png

This file SHOULD NOT contain modeling or preprocessing logic.
"""

import prepare


def main():
    # --------------------------------------------------------
    # 1. Load data (frozen)
    # --------------------------------------------------------
    # Returns:
    #   df_train : pd.DataFrame
    #   df_eval  : pd.DataFrame
    df_train, df_eval = prepare.load_data()

    # --------------------------------------------------------
    # 2. Evaluate current model (from model.py)
    # --------------------------------------------------------
    # Returns validation ROC-AUC
    val_auc = prepare.evaluate(df_train, df_eval)

    # --------------------------------------------------------
    # 3. Log results
    # --------------------------------------------------------
    prepare.log_result(
        experiment_id="baseline",
        val_auc=val_auc,
        status="baseline",
        description="Wind speed logistic regression baseline"
    )

    # --------------------------------------------------------
    # 4. Update performance plot
    # --------------------------------------------------------
    prepare.plot_results()

    print(f"✅ Run completed. Validation ROC-AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()