from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name, results_path="results/"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    results_df = pd.DataFrame(report).transpose()
    results_df.to_csv(f"{results_path}{model_name}_results.csv")

    # Save confusion matrix plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{results_path}/{model_name}_cm.png")

    return acc, report, cm

