import torch
import torch.nn as nn

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MoCoLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, logits, labels=None, queue_labels=None):
        """
        logits: Nx(1+K)
        labels: N,
        queue_labels: K,
        """
        device = (torch.device('cuda')
                  if logits.is_cuda
                  else torch.device('cpu'))
        # CL loss
        bsz = logits.shape[0]
        if labels is None and queue_labels is None:
            mask = torch.zeros_like(logits)
            mask[:, 0] = 1.
        else:
            labels = labels.contiguous().view(-1, 1)
            queue_labels = queue_labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, queue_labels.T).float().to(device)  # NxK
            mask = torch.cat([torch.ones(bsz, 1).to(device), mask], dim=1)  # Nx(K+1)

        logits /= self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class SymNegCosineSimilarityLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _neg_cosine_simililarity(self, x, y):
        v = -torch.nn.functional.cosine_similarity(x, y.detach(), dim=-1).mean()
        return v

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        """Forward pass through Symmetric Loss.
        Args:
            out0:
                Output projections of the first set of transformed images.
                Expects the tuple to be of the form (z0, p0), where z0 is
                the output of the backbone and projection mlp, and p0 is the
                output of the prediction head.
            out1:
                Output projections of the second set of transformed images.
                Expects the tuple to be of the form (z1, p1), where z1 is
                the output of the backbone and projection mlp, and p1 is the
                output of the prediction head.
        Returns:
            Contrastive Cross Entropy Loss value.
        Raises:
            ValueError if shape of output is not multiple of batch_size.
        """
        z0, p0 = out0
        z1, p1 = out1

        loss = (
            self._neg_cosine_simililarity(p0, z1) / 2
            + self._neg_cosine_simililarity(p1, z0) / 2
        )

        return loss