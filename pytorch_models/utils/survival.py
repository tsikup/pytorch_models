import torch
from lifelines.statistics import logrank_test
from pycox.models.loss import DeepHitSingleLoss, nll_pmf, _Loss
from torch import Tensor, nn
from torchmetrics import Metric


"""
Continuous Time Survival
"""
#######################
# Functional Survival #
#######################
def coxloss(survtime, event, hazard_pred):
    """
    A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only supports survival data with no ties (i.e., event occurrence at same time).

    The loss functions requires batched input to work properly (i.e. batch size > 1)
    :param survtime:
    :param event:
    :param hazard_pred:
    :return:
    """
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = torch.zeros(size=[current_batch_len, current_batch_len], dtype=torch.float)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(hazard_pred.device)
    event = event.reshape(-1)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * event
    assert event.sum() > 0, "All samples are censored!"
    loss_cox = -(
        loss_cox.sum() / event.sum()
    )  # mean over samples who experienced the event only (where event==1)
    return loss_cox


def accuracy(output, labels, is_update=False):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if is_update:
        return correct, len(labels)
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels, threshold="median", is_update=False):
    # This accuracy is based on estimated survival events against true survival events
    if threshold == "median":
        threshold = torch.median(hazardsdata)
    elif threshold == "mean":
        threshold = torch.mean(hazardsdata)
    else:
        assert isinstance(threshold, float)
    hazards_dichotomize = torch.zeros([len(hazardsdata)], dtype=torch.uint8).to(
        hazardsdata.device
    )
    hazards_dichotomize[hazardsdata > threshold] = 1
    correct = torch.sum(hazards_dichotomize == labels)
    if is_update:
        return correct, len(labels)
    return correct / len(labels)


def cox_log_rank(
    hazardsdata, labels, survtime_all, threshold="median", is_update=False
):
    if threshold == "median":
        threshold = torch.median(hazardsdata)
    elif threshold == "mean":
        threshold = torch.mean(hazardsdata)
    else:
        assert isinstance(threshold, float)
    hazards_dichotomize = torch.zeros([len(hazardsdata)], dtype=torch.uint8)
    hazards_dichotomize[hazardsdata > threshold] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx].tolist()
    T2 = survtime_all[~idx].tolist()
    E1 = labels[idx].tolist()
    E2 = labels[~idx].tolist()
    if is_update:
        return E1, E2, T1, T2
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred


def cindex(hazards, events, survtimes, is_update=False):
    concord = 0.0
    total = 0.0
    N_test = events.shape[0]
    for i in range(N_test):
        if events[i] == 1:
            for j in range(N_test):
                if survtimes[j] > survtimes[i]:
                    total += 1
                    if hazards[j] < hazards[i]:
                        concord += 1
                    elif hazards[j] == hazards[i]:
                        concord += 0.5
    if is_update:
        return concord, total
    return concord / total


def cindex_lifeline(hazards, events, survtime):
    from lifelines.utils import concordance_index

    return concordance_index(survtime, -hazards, events)


def cindex_sksurv(hazards, events, survtime):
    from sksurv.metrics import concordance_index_censored

    return concordance_index_censored(events == 1, survtime, hazards)[0]


#####################
# TorchMetric Utils #
#####################
class AccuracyCox(Metric):
    """
    Accuracy for survival model
    :arg
        hazards: Tensor, shape (N,1) or (N); predicted hazards (logits)
        events: Tensor, shape (N,1) or (N); event status (0 or 1)
        survtimes: Tensor, shape (N,1) or (N); survival time
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, hazards: Tensor, events: Tensor, survtimes: Tensor):
        hazards = hazards.squeeze()
        events = events.squeeze()
        survtimes = survtimes.squeeze()
        assert hazards.shape == events.shape == survtimes.shape
        hazards, events, survtimes = (
            hazards.reshape(-1),
            events.reshape(-1),
            survtimes.reshape(-1),
        )

        hazards = torch.Tensor(hazards) if not torch.is_tensor(hazards) else hazards
        events = torch.Tensor(events) if not torch.is_tensor(events) else events

        _correct, _total = accuracy_cox(hazards, events, is_update=True)
        self.correct += _correct
        self.total += _total

    def compute(self):
        return self.correct.float() / self.total


class CoxLogRank(Metric):
    """
    Log-rank test for survival model
    :arg
        hazards: Tensor, shape (N,1) or (N); predicted hazards (logits)
        events: Tensor, shape (N,1) or (N); event status (0 or 1)
        survtimes: Tensor, shape (N,1) or (N); survival time
    """

    def __init__(self):
        super().__init__()
        self.add_state("E1", default=[], dist_reduce_fx="cat")
        self.add_state("E2", default=[], dist_reduce_fx="cat")
        self.add_state("T1", default=[], dist_reduce_fx="cat")
        self.add_state("T2", default=[], dist_reduce_fx="cat")

    def update(self, hazards: Tensor, events: Tensor, survtimes: Tensor):
        hazards = hazards.squeeze()
        events = events.squeeze()
        survtimes = survtimes.squeeze()
        assert hazards.shape == events.shape == survtimes.shape
        hazards, events, survtimes = (
            hazards.reshape(-1),
            events.reshape(-1),
            survtimes.reshape(-1),
        )

        E1, E2, T1, T2 = cox_log_rank(hazards, events, survtimes, is_update=True)

        assert isinstance(E1, list)
        self.E1 = self.E1 + E1

        assert isinstance(E2, list)
        self.E2 = self.E2 + E2

        assert isinstance(T1, list)
        self.T1 = self.T1 + T1

        assert isinstance(T2, list)
        self.T2 = self.T2 + T2

    def compute(self):
        results = logrank_test(
            self.T1, self.T2, event_observed_A=self.E1, event_observed_B=self.E2
        )
        pvalue_pred = results.p_value
        return pvalue_pred


class CIndex(Metric):
    """
    C-index for survival model
    :arg
        hazards: Tensor, shape (N,1) or (N); predicted hazards (logits)
        events: Tensor, shape (N,1) or (N); event status (0 or 1)
        survtimes: Tensor, shape (N,1) or (N); survival time
    """

    def __init__(self):
        super().__init__()
        self.add_state("concord", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, hazards: Tensor, events: Tensor, survtimes: Tensor):
        hazards = hazards.squeeze()
        events = events.squeeze()
        survtimes = survtimes.squeeze()
        assert hazards.shape == events.shape == survtimes.shape
        hazards, events, survtimes = (
            hazards.reshape(-1),
            events.reshape(-1),
            survtimes.reshape(-1),
        )

        _concord, _total = cindex(hazards, events, survtimes, is_update=True)
        self.concord += _concord
        self.total += _total

    def compute(self):
        return self.concord / self.total


##########################
# Discrete Time Survival #
##########################
# def continuous_to_discrete_time():


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """A maximum likelihood estimation function in Survival Analysis.

    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...

    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py

    :param hazards: (N,K); hazards for each bin (K) in batch (N)
    :param S: (N,K); cumulative product of 1 - hazards, i.e. survival probability at each bin
    :param Y: (N,1); ground truth bin, 1,2,...,k; i.e. the bin where the event happened
    :param c: (N,1); censorship status, 0 or 1
    :param alpha:
    :param eps:
    :return:
    """
    batch_size = len(Y)
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        Y = Y.argmax(dim=1).view(batch_size, 1)  # ground truth bin, 1,2,...,k
    else:
        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(
            1 - hazards, dim=1
        )  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat(
        [torch.ones_like(c), S], 1
    )  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py

    :param hazards: (N,K); hazards for each bin (K) in batch (N)
    :param S: (N,K); cumulative product of 1 - hazards, i.e. survival probability at each bin
    :param Y: (N,1); ground truth bin, 1,2,...,k; i.e. the bin where the event happened
    :param c: (N,1); censorship status, 0 or 1
    :param alpha:
    :param eps:
    :return:
    """
    batch_size = len(Y)
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        Y = Y.argmax(dim=1).view(batch_size, 1)  # ground truth bin, 1,2,...,k
    else:
        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(
            1 - hazards, dim=1
        )  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps)
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps)
    )
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    """
    Cross entropy loss for survival model
    :arg
        hazards: Tensor, shape (N,1); predicted hazards (logits)
        S: Tensor, shape (N,K); cumulative product of 1 - hazards, i.e. survival probability at each bin
        Y: Tensor, shape (N,1); ground truth bin, 1,2,...,k; i.e. the bin where the event happened
        c: Tensor, shape (N,1); censor status (0 or 1)
    """

    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    """
    Negative log likelihood loss for survival model
    :arg
        hazards: Tensor, shape (N,1); predicted hazards (logits)
        S: Tensor, shape (N,K); cumulative product of 1 - hazards, i.e. survival probability at each bin
        Y: Tensor, shape (N,1); ground truth bin, 1,2,...,k; i.e. the bin where the event happened
        c: Tensor, shape (N,1); censor status (0 or 1)
    """

    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    """
    Cox loss for survival model

    The loss functions requires batched input to work properly (i.e. batch size > 1)
    :arg
        hazards: Tensor, shape (N,1); predicted hazards (logits)
        events: Tensor, shape (N,1); event status (0 or 1)
        survtimes: Tensor, shape (N,1); survival time
    """

    def __call__(self, hazards, survtimes, events, **kwargs):
        return coxloss(survtimes, events, hazards)


class MyDeepHitLoss(nn.Module):
    def __init__(self, alpha=0.5, sigma=1.0):
        super().__init__()
        self.loss = DeepHitSingleLoss(alpha=alpha, sigma=sigma)

    def forward(self, survtimes, events, hazards, **kwargs):
        raise NotImplementedError
        rank_mat = None
        return self.loss.forward(hazards, survtimes, events, rank_mat=rank_mat)


################################################
# Hybrid Continuous and Discrete Time Survival #
################################################
class HybridDeepHitLoss(_Loss):
    """Hybrid DeepHit Loss (ranking loss on continuous time and nll on discrete time)
    loss = alpha * nll + (1 - alpha) * rank_loss

    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.

    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.
    """

    def __init__(self, alpha: float, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"Need `alpha` to be in [0, 1]. Got {alpha}.")
        self._alpha = alpha

    def forward(
        self,
        hazards: Tensor,
        idx_durations: Tensor,
        survtimes: Tensor,
        events: Tensor,
        rank_mat: Tensor,
    ) -> Tensor:
        nll = nll_pmf(hazards, idx_durations, events, self.reduction)
        rank_loss = coxloss(survtimes, events, hazards)
        return self.alpha * nll + (1.0 - self.alpha) * rank_loss
