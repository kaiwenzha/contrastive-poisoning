import torch
import torch.nn as nn
import copy

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MoCo(nn.Module):
    def __init__(self, base_encoder, arch='resnet18', dim=128, K=65536, m=0.999, T=0.07, mlp=False, allow_mmt_grad=False):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.allow_mmt_grad = allow_mmt_grad

        self.encoder_q = base_encoder(arch=arch, head='mlp' if mlp else 'linear', feat_dim=dim)
        self.encoder_k = base_encoder(arch=arch, head='mlp' if mlp else 'linear', feat_dim=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        # compute key features
        with torch.set_grad_enabled(self.allow_mmt_grad):  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if not self.allow_mmt_grad:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC

            # undo shuffle
            if not self.allow_mmt_grad:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k if not self.allow_mmt_grad else k.clone().detach())

        return logits

def _deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters."""
    for param in params:
        param.requires_grad = False


def _do_momentum_update(prev_params, params, m):
    """Updates the weights of the previous parameters."""
    for prev_param, param in zip(prev_params, params):
        prev_param.data = prev_param.data * m + param.data * (1.0 - m)

class _MomentumEncoderMixin:
    m: float
    backbone: nn.Module
    projection_head: nn.Module
    momentum_backbone: nn.Module
    momentum_projection_head: nn.Module

    def _init_momentum_encoder(self):
        """Initializes momentum backbone and a momentum projection head."""
        assert self.backbone is not None
        assert self.projection_head is not None

        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection_head = copy.deepcopy(self.projection_head)

        _deactivate_requires_grad(self.momentum_backbone.parameters())
        _deactivate_requires_grad(self.momentum_projection_head.parameters())

    @torch.no_grad()
    def _momentum_update(self, m: float = 0.999):
        """Performs the momentum update for the backbone and projection head."""
        _do_momentum_update(
            self.momentum_backbone.parameters(),
            self.backbone.parameters(),
            m=m,
        )
        _do_momentum_update(
            self.momentum_projection_head.parameters(),
            self.projection_head.parameters(),
            m=m,
        )

    @torch.no_grad()
    def _batch_shuffle(self, batch: torch.Tensor):
        """Returns the shuffled batch and the indices to undo."""
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle

    @torch.no_grad()
    def _batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        """Returns the unshuffled batch."""
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]

def _get_byol_mlp(num_ftrs, hidden_dim, out_dim):
    modules = [
        nn.Linear(num_ftrs, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    ]
    return nn.Sequential(*modules)

class BYOL(nn.Module, _MomentumEncoderMixin):
    def __init__(self, backbone, num_ftrs=512, hidden_dim=4096, out_dim=256, m=0.999, allow_mmt_grad=False):
        super(BYOL, self).__init__()

        self.backbone = backbone
        self.projection_head = _get_byol_mlp(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = _get_byol_mlp(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m
        self.allow_mmt_grad = allow_mmt_grad

    def _forward(self, x0, x1=None, return_features=False):
        self._momentum_update(self.m)
        # forward pass of first input x0
        f0 = self.backbone(x0).squeeze()
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.set_grad_enabled(self.allow_mmt_grad):

            f1 = self.momentum_backbone(x1).squeeze()
            out1 = self.momentum_projection_head(f1)

            if return_features:
                out1 = (out1, f1)

        return out0, out1

    def forward(self, x0, x1=None, return_features=False):
        p0, z1 = self._forward(x0, x1, return_features=return_features)
        p1, z0 = self._forward(x1, x0, return_features=return_features)

        return (z0, p0), (z1, p1)