def plot_precision_recall_curves(models, X_test, y_test, dataset_type):
    """
    Plots precision-recall curves for multiple models.

    Parameters:
    - models: A dictionary with model names as keys and trained models as values.
    - X_test: The test feature data.
    - y_test: The true labels for the test set.
    - dataset_type: String indicating the dataset type ('TRANSFER' or 'CASH OUT').

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    for model_name, model in models.items():
        # Get predictions and calculate precision-recall curve
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auprc = auc(recall, precision)
        
        # Plot the curve
        plt.plot(recall, precision, label=f"{model_name} - AUPRC: {auprc:.4f}")

    # Add plot details
    plt.title(f"{dataset_type} dataset - Precision-Recall Curve", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid()
    plt.show()