import prepare
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import model

def load_test():
    from processing.tfdata import get_dataset
    from processing.features import dataset_to_dataframe

    test_ds = get_dataset(
        "data/next_day_wildfire_spread_test_*.tfrecord",
        data_size=64,
        sample_size=32,
        batch_size=32,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=True,
    )
    return dataset_to_dataframe(test_ds)

if __name__ == "__main__":
    print("Loading train and eval for fitting...")
    df_train, df_eval = prepare.load_data()

    print("Loading test set...")
    df_test = load_test()

    print(f"Test rows: {len(df_test)}")
    print(f"Test positive rate: {df_test['fire_any'].mean():.4f}")

    print("Running model...")
    y_scores = model.compute_metric(df_train, df_eval)

    # Re-run on test
    y_true = df_test["fire_any"].values

    # Need to refit on train+eval, predict on test
    # So we call compute_metric with test as the eval set
    y_scores_test = model.compute_metric(df_train, df_test)

    print("\n--- Test Results ---")
    print(f"ROC-AUC:  {roc_auc_score(y_true, y_scores_test):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_true, y_scores_test):.4f}")
    print(f"F1:       {f1_score(y_true, (y_scores_test >= 0.5).astype(int)):.4f}")
    print()
    print(classification_report(y_true, (y_scores_test >= 0.5).astype(int),
          target_names=["no fire", "fire"]))