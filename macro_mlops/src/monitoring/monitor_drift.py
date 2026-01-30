from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result = report.as_dict()

    drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    total = result["metrics"][0]["result"]["number_of_columns"]

    drift_ratio = drifted / total

    return {
        "drifted_features": drifted,
        "total_features": total,
        "drift_ratio": drift_ratio,
        "drift_detected": drift_ratio >= 0.3,
    }
