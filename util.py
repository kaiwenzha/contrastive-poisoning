import datetime
import logging
import math
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def set_model_backbone_grad(cl_alg, model, flag):
    if cl_alg == 'SimCLR':
        for param in model.module.backbone.parameters():
            param.requires_grad = flag
    elif cl_alg == 'BYOL':
        for param in model.module.backbone.backbone.parameters():
            param.requires_grad = flag
        for param in model.module.backbone.projection_head.parameters():
            param.requires_grad = flag
        for param in model.module.backbone.prediction_head.parameters():
            param.requires_grad = flag
    else:
        for param in model.module.backbone.encoder_q.parameters():
            param.requires_grad = flag

def convert_classwise_to_samplewise(classwise_noise, opt):
    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.data_folder)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=opt.data_folder)
    dataset_size = dataset.__len__()
    N, C, H, W = classwise_noise.shape
    samplewise_noise = torch.zeros(dataset_size, C, H, W)
    for i in range(dataset_size):
        samplewise_noise[i] = classwise_noise[dataset.targets[i]]
    return samplewise_noise

def save_model(model, optimizer, delta_optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'delta_optimizer': delta_optimizer.state_dict() if delta_optimizer is not None else None,
        'epoch': epoch,
    }

    torch.save(state, save_file)

    del state

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class TextFormat:
    ColorCode = {
        'black':        '\033[30m',
        'darkred':      '\033[31m',
        'darkgreen':    '\033[32m',
        'darkyellow':   '\033[33m',
        'darkblue':     '\033[34m',
        'darkpink':     '\033[35m',
        'darkcyan':     '\033[36m',
        'grey':         '\033[37m',
        'white':        '\033[38m',
        'darkgrey':     '\033[90m',
        'red':          '\033[91m',
        'green':        '\033[92m',
        'yellow':       '\033[93m',
        'blue':         '\033[94m',
        'pink':         '\033[95m',
        'cyan':         '\033[96m',
    }
    StyleCode = {
        'normal':        '\033[0m',
        'bold':          '\033[01m',
        'disable':       '\033[02m',
        'underline':     '\033[04m',
        'reverse':       '\033[07m',
        'strikethrough': '\033[09m',
        'invisible':     '\033[08m',
    }
    EndCode = '\033[0m'

    @classmethod
    def format(cls, text, color='white'):
        return cls.ColorCode[color] + text + cls.EndCode

def log(text, color='white', style='normal', with_time=True, handle=None):
    if with_time:
        text = '[' + datetime.datetime.now().strftime('%H:%M:%S') + '] ' + str(text)
    logging.info(TextFormat.StyleCode[style] + TextFormat.ColorCode[color] + str(text) + TextFormat.EndCode)
    if handle is not None:
        handle.write(str(text) + '\n')
    return text

def concat_all_gather(tensor):
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    output = torch.cat(output, dim=0)

    return output

class GatherLayer(torch.autograd.Function):
    '''
    Gather tensors from all process, supporting backward propagation.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out