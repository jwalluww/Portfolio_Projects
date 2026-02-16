from evidently import Report
from evidently.presets import DataDriftPreset

def check_drift(reference_df, current_df):
    # Set up a template for the report with the DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])

    # Run the report using a reference dataset and a current dataset
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    # Extract drift metrics from the report dictionary
    drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    total = result["metrics"][0]["result"]["number_of_columns"]

    drift_ratio = drifted / total

    return {
        "drifted_features": drifted, # Count of features drifted
        "total_features": total, # Total number of features
        "drift_ratio": drift_ratio, # Ratio of drifted features to total features
        "drift_detected": drift_ratio >= 0.3, # Boolean indicating if drift is detected based on arbitrary threshold
    }