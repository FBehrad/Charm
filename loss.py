import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


class Metrics:
    def __init__(self, data):
        self.dataset = data
        pass

    def calculate(self, preds, labels):
        srcc, _ = spearmanr(preds, labels)
        plcc, _ = pearsonr(preds, labels)

        if self.dataset == 'ava':
            threshold = 5
        elif self.dataset == 'para':
            threshold = 3
        elif self.dataset == 'tad66k':
            threshold = 0.5
        else:
            threshold = 0.5
        binary_preds = np.array(preds) > threshold
        binary_labels = np.array(labels) > threshold
        accuracy = accuracy_score(binary_preds, binary_labels)

        mse = mean_squared_error(preds, labels)
        mae = mean_absolute_error(preds, labels)

        return plcc, srcc, accuracy, mse, mae


class EMDLoss(nn.Module):
    def __init__(self, r=2, weighted=False):
        super(EMDLoss, self).__init__()
        self.weighted = weighted
        self.r = r

    def single_emd_loss(self, p, q, r=2):
        """
        Earth Mover's Distance of one sample

        Args:
            p: true distribution of shape num_classes × 1
            q: estimated distribution of shape num_classes × 1
            r: norm parameter
        """
        assert p.shape == q.shape, "Length of the two distribution must be the same"
        length = p.shape[0]
        emd_loss = 0.0
        predictions_cumsum = torch.cumsum(q, dim=-1)
        ratings_cumsum = torch.cumsum(p, dim=-1)
        squared_diff = (predictions_cumsum - ratings_cumsum)**2
        emd = torch.sqrt(squared_diff.mean(-1))

        # for i in range(1, length + 1):
        #     emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
        return emd  # (emd_loss / length) ** (1. / r)

    def forward(self, labels, predictions):
        assert labels.shape == predictions.shape, "Shape of the two distribution batches must be the same."
        mini_batch_size = labels.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(self.single_emd_loss(labels[i], predictions[i], r=self.r))

        final_loss = sum(loss_vector) / mini_batch_size

        return final_loss


class MSE_Loss(nn.Module):
    def __init__(self, weighted=False):
        super(MSE_Loss, self).__init__()
        self.weighted = weighted

    def forward(self, predictions, labels, weight=None):
        if self.weighted:
            loss = torch.sum(weight * (predictions - labels) ** 2)
        else:
            loss = torch.sum((predictions - labels) ** 2)
        return loss


def prepare_loss(dataset):
    if dataset == 'ava' or dataset == 'para':
        loss = EMDLoss(r=2)
    else:
        loss = nn.MSELoss()
    return loss


class WeightedLoss(torch.nn.Module):
    def __init__(self, criterion, agreement_weights):
        super(WeightedLoss, self).__init__()
        self.criterion = criterion
        self.agreement_weights = agreement_weights

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        weights = self.agreement_weights.unsqueeze(1).expand_as(loss)
        weighted_loss = weights * loss
        return weighted_loss.mean()




