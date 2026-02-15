import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def evaluate_model(model, test_generator):

    preds = model.predict(test_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, preds[:, 1])
    roc_auc = auc(fpr, tpr)

    print("ROC-AUC:", roc_auc)

    return roc_auc
