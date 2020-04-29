import torch

from numbers import Number


def generalized_binomial_coefficient(logits, k_max, intermediate=None):
    r'''Calculate the generalized binomial coefficient for log-odds

    For a set of odds :math:`w_t \in [0, 1]` and :math:`t \in [1, T]`, the
    generalized binomial coefficient for some number of choices :math:`k` is
    defined as

    .. math::

        C(k, w) = \sum_{\{S \subseteq T: |S| = k\}} \prod_{t \in S} w_t

    This function computes :math:`log C(k, w)` using
    :math:`\text{logits}_t = \log w_t`. `logits` is a float vector of shape
    ``(T,)`` or a matrix of shape ``(T, N)``, where ``T`` is the choice
    dimension and ``N`` is the batch dimension. `k_max` is an integer of the
    maximum `k` value to compute.

    If `intermediate` is :obj:`None` (the default), this function returns `lC`,
    which is either a vector of shape ``(k_max + 1,)`` or a matrix of shape
    ``(k_max + 1, N)``, depending on whether `logits` was batched.
    :math:`lC[k] = \log C(k, w)`.

    If `intermediate` is either ``"forward"`` or ``"reverse"``, `Cl` is a
    tensor of shape ``(T + 1, k.max() + 1[, N])`` that stores all intermediate
    values used in the computation. If ``"forward"``,
    :math:`lC[t, k'] = \log C(k', w_{\leq t})`. If ``"reverse"``, the mapping
    :math:`w_t \mapsto \tilde{w}_{T - t + 1}` is first applied before the
    forward procedure.

    Parameters
    ----------
    logits : torch.tensor
    k_max : int
    intermediate : {None, "forward", "reverse"}

    Returns
    -------
    lC : torch.tensor
    '''
    # Implementation note: this is the "direct" method from
    # Chen and Liu, 1997. "Statistical applications of the Poisson-Binomial and
    # Conditional Bernoulli distributions"

    if logits.dim() == 1:
        flatten = True
        logits = logits.unsqueeze(-1)
    else:
        flatten = False
    if logits.dim() != 2:
        raise RuntimeError('logits must be one or two dimensional')
    T, N = logits.shape

    if k_max < 0:
        raise RuntimeError('k_max must be >= 0 (note: log C(<0, *) = -inf)')

    if intermediate not in {None, "forward", "reverse"}:
        raise RuntimeError(
            "intermediate must be one of None, 'forward', or 'reverse'")

    if not k_max or not T:
        Cl = torch.full(
            (1, N), 0 if not k_max else -float('inf'),
            device=logits.device, dtype=logits.dtype)
        if intermediate:
            Cl = Cl.expand(T + 1, 1, N)
        if flatten:
            Cl = Cl.squeeze(1)
        return Cl

    lC_first = torch.zeros(1, N, device=logits.device, dtype=logits.dtype)
    lC_rest = torch.full(
        (k_max, N), -float('inf'), device=logits.device, dtype=logits.dtype)
    if intermediate:
        hist = [torch.cat([lC_first, lC_rest])]

    logits = logits.unsqueeze(1)
    for t in range(T):
        if intermediate == 'reverse':
            t = T - t - 1

        x = torch.cat([lC_first, lC_rest[:-1]], 0)
        # if x is infinite or logits is infinite, we don't want a gradient
        x = torch.where(
            torch.isfinite(x + logits[t]),
            x + logits[t],
            x.detach() + logits[t].detach()
        )
        lC_rest = torch.logsumexp(torch.stack([lC_rest, x], -1), -1)
        del x

        if intermediate:
            hist.append(torch.cat([lC_first, lC_rest]))

    if intermediate:
        Cl = torch.stack(hist, 0)
    else:
        Cl = torch.cat([lC_first, lC_rest])
    if flatten:
        Cl = Cl.squeeze(-1)
    return Cl


def poisson_binomial(probs):
    '''Sample a Poisson Binomial

    The Poisson Binomial is a generalization of the Binomial where the
    individual Bernoulli trials need not have equal probability.

    Parameters
    ----------
    probs : torch.Tensor
        A float tensor containing the probability of each Bernoulli trial.
        Of shape ``(T, *)``, where ``T`` is the trial dimension (is
        accumulated).

    Returns
    -------
    b : torch.Tensor
        A float tensor of shape ``(*)`` containing the sum of
        the high Bernoulli trials (between :math:`[0, T]` inclusive).

    See Also
    --------
    PoissonBinomial
    '''
    with torch.no_grad():
        if not probs.dim():
            raise RuntimeError('probs must be at least one dimensional')
        b = torch.bernoulli(probs)
        b = b.sum(-1)

    return b


class PoissonBinomial(torch.distributions.Binomial):
    r'''A Binomial distribution with possibly unequal probabilities per trial

    The Poisson-Binomial distribution is a generalization of the Binomial
    distribution to Bernoulli trials with (possibly) unequal probabilities.
    The probability of :math:`k` trials being high is

    .. math::

        P(K=k) = \frac{C(k, w)}{\prod_t (1 + w_t)}

    where :math:`w_t = \text{probs}_t / (1 - \text{probs}_t) =
    \exp(\text{logits}_t)` are the odds of the :math:`t`-th Bernoulli trial.

    Either `probs` or `logits` must be specified, but not both. Each has a
    shape ``(*, T)``, where ``T`` is the dimension along which the trials
    will be summed.

    Parameters
    ----------
    total_count : None or int or torch.Tensor
        Specifies some number between :math:`[0, T]`` inclusive such that the
        values indexed past this value along the ``T`` dimension will be
        considered padding (right-padding). If unspecified, it will be assumed
        to be the full length ``T``. Otherwise, it must broadcast with either
        ``probs[..., -1]`` or ``logits[..., -1]``.
    probs : torch.Tensor
    logits : torch.Tensor
    '''

    arg_constraints = {
        'total_count': torch.distributions.constraints.dependent,
        'probs': torch.distributions.constraints.unit_interval,
        'logits': torch.distributions.constraints.real,
    }

    def __init__(
            self, total_count=None, probs=None, logits=None,
            validate_args=None):
        # modified version of torch.distributions.Binomial. Make sure to handle
        # license!
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both")
        self._param = probs if logits is None else logits
        is_scalar = isinstance(self._param, Number) or not self._param.dim()
        self._T = 1 if is_scalar else self._param.shape[-1]
        if total_count is None:
            total_count = self._T
            do_mask = False
        else:
            do_mask = True
        if is_scalar:
            self.total_count, self._param = (
                torch.distributions.utils.broadcast_all(
                    total_count, self._param))
            batch_shape = torch.Size()
        else:
            self.total_count, param_wo_T = (
                torch.distributions.utils.broadcast_all(
                    total_count, self._param[..., 0]))
            _, self._param = (
                torch.distributions.utils.broadcast_all(
                    param_wo_T.unsqueeze(-1), self._param))
            batch_shape = param_wo_T.shape
        self.total_count = self.total_count.type_as(self._param)
        # we get rid of padded values right away so we never have to
        # worry about counts > total_count. This'll stop any gradients, too
        if do_mask:
            mask = torch.arange(
                self._T, dtype=self._param.dtype, device=self._param.device)
            mask = self.total_count.unsqueeze(-1) <= mask
        if probs is None:
            if do_mask:
                self._param = self._param.masked_fill(mask, -float('inf'))
            self.logits = self._param
        else:
            if do_mask:
                self._param = self._param.masked_fill(mask, 0.)
            self.probs = self._param
        super(torch.distributions.Binomial, self).__init__(
            batch_shape, validate_args=validate_args)
        if (
                self._validate_args and
                not self.total_count.le(self._T).all() and
                not self.total_count.ge(0)):
            raise ValueError(
                'total_count must be <= first dimension of probs/logits and '
                '>= 0')

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PoissonBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._T,))
        all_log_probs_shape = batch_shape + torch.Size((self._T + 1,))
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = self._param
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = self._param
        if 'log_pmf' in self.__dict__:
            new.log_pmf = self.log_pmf.expand(all_log_probs_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new._T = self._T
        super(torch.distributions.Binomial, new).__init__(
            batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @torch.distributions.utils.lazy_property
    def logits(self):
        # clamp "real" probabilities, but pad those greater than total_count
        logits = torch.distributions.utils.probs_to_logits(self._param, True)
        mask = torch.arange(
            self._T, dtype=self._param.dtype, device=self._param.device)
        mask = self.total_count.unsqueeze(-1) <= mask
        logits.masked_fill_(mask, -float('inf'))
        return logits

    @property
    def mean(self):
        return self.probs.sum(0)

    @property
    def variance(self):
        return (self.probs * (1 - self.probs)).sum(0)

    @torch.distributions.utils.lazy_property
    def log_pmf(self):
        # unfortunately, the generalized binomial coefficient goes T first,
        # whereas we go T last. We need the latter so that we can expand
        # without copying, so we do a bunch of transposition
        logits = self.logits
        if logits.dim() > 1:
            logits = logits.flatten(end_dim=-2).T
        lC = generalized_binomial_coefficient(logits, self._T)
        return lC.T.log_softmax(-1).view(*self._batch_shape, self._T + 1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            shape = self._extended_shape(sample_shape) + torch.Size((self._T,))
            probs = self.probs.expand(shape)
            return poisson_binomial(probs)

    def log_prob(self, value):
        # from torch.distributions.Categorical. Make sure to handle license!
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.log_pmf)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)
