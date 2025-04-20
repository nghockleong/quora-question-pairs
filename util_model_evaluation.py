import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def get_model_evaluation(y_true, y_pred, labels, model_name):
    confusionMatrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=labels)
    cm_display.plot(values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    accuracy, recall, precision, f1 = \
        round(accuracy_score(y_true=y_true, y_pred=y_pred), 3), \
        round(recall_score(y_true=y_true, y_pred=y_pred), 3), \
        round(precision_score(y_true=y_true, y_pred=y_pred), 3), \
        round(f1_score(y_true=y_true, y_pred=y_pred), 3)
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=labels))
    print(f"Accuracy: {accuracy}" )
    print(f"Recall: {recall}" )
    print(f"Precision: {precision}" )
    print(f"F1-score: {f1}" )
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }