import torch
import os, json, random, string
import torch.distributed as dist
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, ExponentialLR, ChainedScheduler

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def args_init(args, rank, world_size):
    r''' Initialize seed and exp_name.
    '''
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if (rank == 0): # master
        print(args)
        makedir(f"{args.logdir}/{args.exp_name}")
        d = args.__dict__
        d['world_size'] = world_size
        with open(f"{args.logdir}/{args.exp_name}/args.json", 'w') as f:
            json.dump(d, f, indent=4)

def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt



class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    Reference:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    @property
    def _warmup_lr(self):
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * (self.last_epoch + 1) / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch - 1:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return self._warmup_lr

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch==None else epoch
        if self.last_epoch >= self.warmup_epoch - 1:
            if not self.finished:
                warmup_lr = [base_lr * self.multiplier for base_lr in self.base_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
                self.finished = True
                return
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)
            return

        for param_group, lr in zip(self.optimizer.param_groups, self._warmup_lr):
            param_group['lr'] = lr

    def step(self, metrics=None, epoch=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        result = {key: value for key, value in self.__dict__.items() if key != 'optimizer' or key != "after_scheduler"}
        if self.after_scheduler:
            result.update({"after_scheduler": self.after_scheduler.state_dict()})
        return result

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop("after_scheduler", None)
        self.__dict__.update(state_dict)
        if after_scheduler_state:
            self.after_scheduler.load_state_dict(after_scheduler_state)


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def buildROC(labels, score, targetEff=[0.3,0.5]):
    r''' ROC curve is a plot of the true positive rate (Sensitivity) in the function of the false positive rate
    (100-Specificity) for different cut-off points of a parameter. Each point on the ROC curve represents a
    sensitivity/specificity pair corresponding to a particular decision threshold. The Area Under the ROC
    curve (AUC) is a measure of how well a parameter can distinguish between two diagnostic groups.
    '''
    if not isinstance(targetEff, list):
        targetEff = [targetEff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in targetEff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS

class MMDScheduler:

    def __init__(self, turnon, width, coef=1):
        self.turnon=turnon
        self.width=width
        self.coef = coef

    def __call__(self, epoch):
        if self.width>0:
            return self.coef*(1+torch.tanh((torch.tensor(epoch)-self.turnon)/self.width))/2
        elif self.width==0:
            return self.coef*(epoch>=self.turnon)
        else:
            print('MMD scheduler width is not allowed to be negative')

def make_chained(optimizer, factors, epochs):
    const_sched = ConstantLR(optimizer, factor=factors[0], total_iters=epochs)
    gamma_val = (factors[1]/factors[0])**(1/(epochs-1))
    exp_sched = ExponentialLR(optimizer, gamma=gamma_val)
    chained_scheduler = ChainedScheduler([const_sched, exp_sched])
    #chained_scheduler.last_epoch=-1
    return chained_scheduler

class LinearLambda:

    def __init__(self, rates, epochs):
        self.slope = (rates[1]-rates[0])/(epochs[1]-epochs[0])
        self.intercept = rates[0] - self.slope*epochs[0]

    def __call__(self, epoch):
        return self.slope*epoch + self.intercept 

class ExpLambda:

    def __init__(self, rates, epochs):
        self.factor = (rates[1]/rates[0])**(1/(epochs[1]-epochs[0]))
        self.start_rate = rates[0]
        self.start_epoch = epochs[0]

    def __call__(self, epoch):
        return self.start_rate*self.factor**(epoch-self.start_epoch)

class ParticleNetLambda:

    def __init__(self, rates, epochs, types):
        self.lambdas = [LinearLambda(r, e) if t=='linear' else ExpLambda(r, e) for r, e, t in zip(rates, epochs, types)]
        self.epochs = epochs
    
    def __call__(self, epoch):
        epoch = float(epoch)
        condlist = [epoch>=e[0] for e in self.epochs]
        return float(np.piecewise(epoch, condlist, self.lambdas))
