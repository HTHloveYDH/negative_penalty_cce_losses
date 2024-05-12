import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativePenaltyCategoricalCrossentropy(nn.Module):
    def __init__(self, class_num:int, p_indices:list, alpha=1.0, penalty_scale=None, reduction='mean', \
                 from_where='softmax', eps=1e-10, name='negative_penalty_categorical_crossentropy'):
        super(NegativePenaltyCategoricalCrossentropy, self).__init__()
        self.p_indices = [[p_index] for p_index in p_indices]
        self.alpha = alpha
        self.penalty_scale = float(len(p_indices)) if penalty_scale is None else penalty_scale
        self.penalty_label = _get_penalty_label(class_num, p_indices)
        self.reduction_fn = {
            'none': _no_reduction_over_batch, 'mean': _average_over_batch, 
            'sum': _sum_over_batch
        }[reduction]
        self.cce_loss_fn = {
            'logits': _cce_loss_from_logits, 'softmax': _cce_loss_from_softmax, 
        }[from_where]
        self.penalty_loss_fn = {
            'logits': _penalty_loss_from_logits, 'softmax': _penalty_loss_from_softmax, 
        }[from_where]
        self.eps = eps

    def forward(self, y_pred, y_true):
        losses = _get_losses(
            y_true, y_pred, self.p_indices, self.penalty_label, self.alpha, self.penalty_scale, self.eps,
            self.cce_loss_fn, self.penalty_loss_fn
        )
        losses = self.reduction_fn(losses)
        return losses


class NegativePenaltySparseCategoricalCrossentropy(nn.Module):
    def __init__(self, class_num:int, p_indices:list, alpha=1.0, penalty_scale=None, reduction='mean', \
                 from_where='softmax', eps=1e-10, name='negative_penalty_sparse_categorical_crossentropy'):
        super(NegativePenaltySparseCategoricalCrossentropy, self).__init__()
        self.p_indices = [[p_index] for p_index in p_indices]
        self.alpha = alpha
        self.penalty_scale = float(len(p_indices)) if penalty_scale is None else penalty_scale
        self.penalty_label = _get_penalty_label(class_num, p_indices)
        self.reduction_fn = {
            'none': _no_reduction_over_batch, 'mean': _average_over_batch, 
            'sum': _sum_over_batch
        }[reduction]
        self.cce_loss_fn = {
            'logits': _cce_loss_from_logits, 'softmax': _cce_loss_from_softmax, 
        }[from_where]
        self.penalty_loss_fn = {
            'logits': _penalty_loss_from_logits, 'softmax': _penalty_loss_from_softmax, 
        }[from_where]
        self.eps = eps

    def forward(self, y_pred, y_true):
        num_classes = y_pred.shape[-1]
        # y_true = torch.squeeze(F.one_hot(y_true, num_classes=num_classes), dim=1)
        y_true = F.one_hot(y_true, num_classes=num_classes).float()
        losses = _get_losses(
            y_true, y_pred, self.p_indices, self.penalty_label, self.alpha, self.penalty_scale, self.eps,
            self.cce_loss_fn, self.penalty_loss_fn
        )
        losses = self.reduction_fn(losses)
        return losses


def _get_losses(y_true, y_pred, p_indices:list, penalty_label:list, alpha:float, penalty_scale:float, 
                eps:float, cce_loss_fn, penalty_loss_fn):
    batch_size = y_true.shape[0]
    # cce_loss_sample_weights
    cce_loss_sample_weights = torch.any(
        torch.transpose(torch.eq(torch.tensor(p_indices), torch.argmax(y_true, dim=-1)), 0, 1), dim=-1
    ).float()
    # cce loss
    cce_losses = cce_loss_fn(y_pred, y_true, eps)
    cce_losses = cce_loss_sample_weights * cce_losses
    # y_penalty
    y_penalty = torch.repeat_interleave(torch.unsqueeze(torch.tensor(penalty_label), dim=0), batch_size, dim=0)
    # penalty_loss_sample_weights
    penalty_loss_sample_weights = 1.0 - cce_loss_sample_weights
    # penalty loss
    penalty_losses = penalty_loss_fn(y_pred, y_penalty, penalty_scale, eps)
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # total loss
    losses = cce_losses + alpha * penalty_losses
    return losses


def _no_reduction_over_batch(losses):
    return losses


def _average_over_batch(losses):
    return torch.mean(losses)


def _sum_over_batch(losses):
    return torch.sum(losses)


def _cce_loss_from_logits(y_pred, y_true, eps):
    return F.cross_entropy(y_pred, y_true, reduction='none')


def _cce_loss_from_softmax(y_pred, y_true, eps):
    return torch.sum(-y_true * torch.log(torch.clip(y_pred, eps, 1.0 - eps)), dim=-1)


def _penalty_loss_from_logits(y_pred, y_penalty, penalty_scale, eps):
    return F.cross_entropy(1.0 - y_pred, y_penalty, reduction='none') / penalty_scale


def _penalty_loss_from_softmax(y_pred, y_penalty, penalty_scale, eps):
    return torch.sum(
        -y_penalty * torch.log(torch.clip(1.0 - y_pred, eps, 1.0 - eps)), dim=-1
    ) / penalty_scale


def _get_penalty_label(class_num:int, p_indices:list):
    penalty_label = [1 if i in p_indices else 0 for i in range(0, class_num)]
    return penalty_label
