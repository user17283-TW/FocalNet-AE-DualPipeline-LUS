import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score,\
      f1_score, recall_score, precision_score

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., use_gpu=True):
        self.std = std
        self.mean = mean
        self.use_gpu = use_gpu
        
    def __call__(self, tensor: torch.Tensor):
        device = tensor.device if self.use_gpu else 'cpu'
        noise = torch.randn(tensor.size(), device=device) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def calc_metrics(y_true, y_pred, auc=None, comment=None, verbose=True):
    """
    Calculate the metrics of the model
    """

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1_score_result = f1_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    weighted_accuracy = balanced_accuracy_score(y_true, y_pred)

    if(verbose):
        print(f"Accuracy: {accuracy}")
        print('Weighted accuracy: ', weighted_accuracy)
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"Specificity: {specificity}")
        print(f"F1 Score: {f1_score_result}")

    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1_score": f1_score_result,
        "weighted_accuracy": weighted_accuracy
    }

    if(comment is not None):
        metrics["comment"] = comment

    if(auc is not None):
        metrics["auc"] = auc

    return metrics
