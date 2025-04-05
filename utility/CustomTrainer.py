import os
import numpy as np
import evaluate
from transformers import (
     TrainingArguments,
     Trainer,
     logging,
)
import torch
from ray import tune, train

class CustomTrainer(Trainer):
    """
    CustomTrainer
    
    In order to use custom loss function we need to create a subclass of Trainer that inherits from HuggingFace's Trainer class.
    
    Note: Soft MCC and F1 only work with a binary classification task, but can be amended to work with multi-class classification tasks by 
          aggregating one-vs-all metrics.
    
    Args (only those alterd/added on top of parent class Trainer):
        type_loss (String): Name of loss function to be used during model training/optimization.
        loss_fcts (Dictionary): Maps name of loss functions to respective methods.
        class_weights (List): Contain individual class weights to be used when employing weighted loss function schemes.
        kwargs: Default Trainer attributes. For details see: https://huggingface.co/docs/transformers/main_classes/trainer
    
    Methods (only those alterd/added on top of parent class Trainer):
    """

    def __init__(self, type_loss, class_weights, **kwargs):
        # Instantiate Parent Class
        super().__init__(**kwargs)
        # Assign ChildClass Attributes
        self.type_loss = type_loss
        self.loss_fcts = {"wce": self.weighted_cross_entropy, "f1": self.soft_f1, "mcc": self.soft_mcc, "ce":super().compute_loss}
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overwrite parent's compute_loss, this function will return the desired loss for
        function specified during initialization of class.
        """
        return self.loss_fcts[self.type_loss](model, inputs, return_outputs)

    def soft_f1(self, model, inputs, return_outputs=False):

        """
        Compute soft F1-score as a cost function (1 - average soft-F1 across all labels).
        
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
        https://arxiv.org/abs/2108.10566
        https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
        """
        
        # prepare inputs
        y = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        y_hat = torch.nn.functional.softmax(logits, dim=1)
        # construct soft scores
        tp = (y_hat[:, 1] * y).sum(dim=0)
        fn = (y_hat[:, 0] * y).sum(dim=0)
        fp = (y_hat[:, 1] * (1-y)).sum(dim=0)
        tn = (y_hat[:, 0] * (1-y)).sum(dim=0)
        # calculate cost
        soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
        # reduce 1 - f1 to maximize f1
        cost_class1 = 1 - soft_f1_class1
        cost_class0 = 1 - soft_f1_class0 
        cost = 0.5 * (cost_class1 + cost_class0)
        # compute average
        loss = cost.mean()
        return (loss, outputs) if return_outputs else loss

    def weighted_cross_entropy(self, model, inputs, return_outputs=False):
        """
        This method employs Cross-Entropy but puts different weights on each
        class.
        With this, should one class be of more importance we can overweigh its
        impact on the loss, thereby indirectly penalizing the other class.
        """
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute loss - adjust weights for classes as desired
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def soft_mcc(self, model, inputs, return_outputs=False):
        """
        Computes a soft MCC score, similiarly to the soft-F1, by using
        probability measures instead of binary predictions.
    
        https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
        https://github.com/vlainic/matthews-correlation-coefficient/tree/master
        https://arxiv.org/abs/2010.13454
        """
        # prepare inputs
        y = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        y_hat = torch.nn.functional.softmax(logits, dim=1)
        # construct soft scores
        tp = (y_hat[:, 1] * y).sum(dim=0)
        fn = (y_hat[:, 0] * y).sum(dim=0)
        fp = (y_hat[:, 1] * (1-y)).sum(dim=0)
        tn = (y_hat[:, 0] * (1-y)).sum(dim=0)
      # calculate cost
        mcc = (tn * tp - fn * fp)/ torch.sqrt(((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))+ 1e-16)
        loss_mcc = 1 - mcc
    
        return (loss_mcc, outputs) if return_outputs else loss_mcc