from torchmetrics import (
    F1Score,
    R2Score,
    PearsonCorrCoef,
    SpearmanCorrCoef,
    HammingDistance,
    AUROC,
    Accuracy,
    Precision,
    Recall,
)

from torch import nn


def psi_pred_metrics(y_preds, y_true, device="cpu"):

    pearson_op = PearsonCorrCoef().to(device)
    pearson_val = pearson_op(y_preds.flatten(), y_true.flatten())

    r2_op = R2Score().to(device)
    r2_val = r2_op(y_preds.flatten(), y_true.flatten())

    spearman_op = SpearmanCorrCoef().to(device)
    spearman_val = spearman_op(y_preds.flatten(), y_true.flatten())

    l1_loss = nn.L1Loss()(y_preds, y_true)
    metric_dict = {
        "pearson_val": pearson_val,
        "r2_val": r2_val,
        "spearman_val": spearman_val,
        "l1_loss": l1_loss.item(),
    }

    return metric_dict


def psi_classification_metrics(y_preds, y_true, num_classes=2, task='binary', device="cpu"):
    """

    y_preds = Nx1 probs
    y_true = Nx1 binary

    """
    if task == 'binary':
        accuracy_op = Accuracy(task=task).to(device)
        auroc_op = AUROC(task=task).to(device)
        precision_op = Precision(task=task).to(device)
        recall_op = Recall(task=task).to(device)

        acc_val = accuracy_op(y_preds.flatten(), y_true.flatten()).item()
        auroc_val = auroc_op(y_preds.flatten(), y_true.flatten()).item()
        precision_val = precision_op(y_preds.flatten(), y_true.flatten()).item()
        recall_val = recall_op(y_preds.flatten(), y_true.flatten()).item()
        
    elif task == 'multiclass':
        # print('multiclass selected')
        accuracy_op = Accuracy(task=task, num_classes=num_classes).to(device)
        auroc_op = AUROC(task=task, num_classes=num_classes).to(device)
        precision_op = Precision(task=task, num_classes=num_classes).to(device)
        recall_op = Recall(task=task, num_classes=num_classes).to(device)


        acc_val = accuracy_op(y_preds.reshape(-1, num_classes), y_true.reshape(-1, 1)).item()
        auroc_val = auroc_op(y_preds.reshape(-1, num_classes), y_true.reshape(-1, 1)).item()
        precision_val = precision_op(y_preds.reshape(-1, num_classes), y_true.reshape(-1, 1)).item()
        recall_val = recall_op(y_preds.reshape(-1, num_classes), y_true.reshape(-1, 1)).item()

    metric_dict = {
        "accuracy": acc_val,
        "AUROC": auroc_val,
        "Precision": precision_val,
        "Recall": recall_val,
    }

    return metric_dict


