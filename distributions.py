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
    total_count : None or int or torch.Tensor, optional
        Specifies some number between :math:`[0, T]`` inclusive such that the
        values indexed past this value along the ``T`` dimension will be
        considered padding (right-padding). If unspecified, it will be assumed
        to be the full length ``T``. Otherwise, it must broadcast with ``(*)``
    probs : torch.Tensor, optional
    logits : torch.Tensor, optional
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
            mask = self._padding_mask()
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

    def _padding_mask(self):
        mask = torch.arange(
                self._T, dtype=self._param.dtype, device=self._param.device)
        mask = self.total_count.unsqueeze(-1) <= mask
        return mask

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
        mask = self._padding_mask()
        logits.masked_fill_(mask, -float('inf'))
        return logits

    @property
    def mean(self):
        return self.probs.sum(-1)

    @property
    def variance(self):
        return (self.probs * (1 - self.probs)).sum(-1)

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


class ConditionalBernoulli(torch.distributions.Bernoulli):
    r'''Bernoulli trials conditioned on the total number of emitting trials

    The Conditional Bernoulli distribution samples Bernoulli trials under the
    condition that the total number of emitting (value 1) trials equals some
    fixed value. Letting :math:`L = \text{condition\_count}`,
    :math:`w_t = \exp(\text{logits}_t) =
    \text{probs}_t / (1 - \text{probs}_t)` and :math:`A \subseteq [1, T],
    |A| = L` be a set of emitting Bernoulli trials, the probability of
    :math:`A` given :math:`L` is:

    .. math::

        P(A|L) = \frac{\prod_{a \in A} w_a}{C(L, w)}

    where :math:`C(L, w)` is the generalized binomial coefficient.

    Either `probs` or `logits` must be specified, but not both. Each has a
    shape ``(*, T)``, where ``T`` is the dimension along which the trials
    are selected to match `condition_count`. `condition_count` must broadcast
    with ``(*)``.

    Parameters
    ----------
    condition_count : int or torch.Tensor
    total_count : None or int or torch.Tensor, optional
        Specifies some number between :math:`[\text{condition\_count}, T]``
        inclusive such that the values indexed past this value along the ``T``
        dimension will be considered padding (right-padding). If unspecified,
        it will be assumed to be the full length ``T``. Otherwise, it must
        broadcast with ``(*)``
    probs : torch.Tensor, optional
    logits : torch.Tensor, optional
    '''

    arg_constraints = {
        'condition_count': torch.distributions.constraints.dependent,
        'total_count': torch.distributions.constraints.dependent,
        'probs': torch.distributions.constraints.unit_interval,
        'logits': torch.distributions.constraints.real_vector,
    }

    class _SampleConstraint(torch.distributions.constraints.Constraint):
        '''Ensures samples could be from this Conditional Bernoulli'''

        def __init__(self, total_count, condition_count):
            self.total_count = total_count
            self.condition_count = condition_count

        def check(self, value):
            is_bool_mask = (value == 0) | (value == 1)
            counts = value * torch.arange(value.shape[-1], device=value.device)
            total_count = self.total_count.expand(counts.shape[:-1])
            total_count_mask = total_count.unsqueeze(-1) > counts
            val_sum = value.sum(-1)
            condition_count = self.condition_count.expand_as(val_sum)
            cond_count_mask = (val_sum == condition_count)
            return (
                is_bool_mask & total_count_mask & cond_count_mask.unsqueeze(-1)
            )

    sample_constraint = _SampleConstraint

    def __init__(
            self, condition_count, total_count=None, probs=None, logits=None,
            validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both")
        self._param = probs if logits is None else logits
        is_scalar = isinstance(self._param, Number) or not self._param.dim()
        self._T = 1 if is_scalar else self._param.shape[-1]
        event_shape = torch.Size((self._T,))
        if total_count is None:
            total_count = self._T
            do_mask = False
        else:
            do_mask = True
        if is_scalar:
            self.total_count, self.condition_count, self._param = (
                torch.distributions.utils.broadcast_all(
                    total_count, condition_count, self._param))
            batch_shape = torch.Size()
        else:
            self.total_count, self.condition_count, param_wo_T = (
                torch.distributions.utils.broadcast_all(
                    total_count, condition_count, self._param[..., 0]))
            _, self._param = (
                torch.distributions.utils.broadcast_all(
                    param_wo_T.unsqueeze(-1), self._param))
            batch_shape = param_wo_T.shape
        self.total_count = self.total_count.type_as(self._param)
        self.condition_count = self.condition_count.type_as(self._param)
        # we get rid of padded values right away so we never have to
        # worry about counts > total_count. This'll stop any gradients, too
        if probs is None:
            if do_mask:
                self._param = self._param.masked_fill(
                    self._padding_mask(), -float('inf'))
            self.logits = self._param
        else:
            if do_mask:
                self._param = self._param.masked_fill(
                    self._padding_mask(), 0.)
            self.probs = self._param
        self._L = int(self.condition_count.max().item())
        super(torch.distributions.Bernoulli, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape,
            validate_args=validate_args)
        if (
                self._validate_args and
                (
                    self.total_count.gt(self._T).any() or
                    self.total_count.lt(self.condition_count).any())):
            raise ValueError(
                'total_count must be <= first dimension of probs/logits and '
                '>= condition_count')

    def _padding_mask(self):
        mask = torch.arange(
                self._T, dtype=self._param.dtype, device=self._param.device)
        mask = self.total_count.unsqueeze(-1) <= mask
        return mask

    @torch.distributions.constraints.dependent_property
    def support(self):
        return self.sample_constraint(self.total_count, self.condition_count)

    @torch.distributions.utils.lazy_property
    def logits(self):
        # clamp "real" probabilities, but pad those greater than total_count
        logits = torch.distributions.utils.probs_to_logits(self._param, True)
        logits.masked_fill_(self._padding_mask(), -float('inf'))
        return logits

    @torch.distributions.utils.lazy_property
    def lC_reverse(self):
        # as with poisson_binomial, we need to push the T dimension last. We
        # have to do the same for the L dimension.
        logits = self.logits
        if logits.dim() > 1:
            logits = logits.flatten(end_dim=-2).T
        lC = generalized_binomial_coefficient(logits, self._L, "reverse")
        lC = lC.flatten(end_dim=1).T  # (N, (T + 1) * (L + 1))
        return lC.reshape(*self._batch_shape, self._T + 1, self._L + 1)

    @torch.distributions.utils.lazy_property
    def lC_forward(self):
        logits = self.logits
        if logits.dim() > 1:
            logits = logits.flatten(end_dim=-2).T
        lC = generalized_binomial_coefficient(logits, self._L, "forward")
        lC = lC.flatten(end_dim=1).T  # (N, (T + 1) * (L + 1))
        return lC.reshape(*self._batch_shape, self._T + 1, self._L + 1)

    @property
    def mean(self):
        # we want the overall probability of w_t being drafted.
        # the probability of being drafted as the l-th term is
        # P(t = t_l) = C(l - 1, <t) w_t C(L - l, >t) / Z
        # and if we sum all l we get
        # P(t) = w_tC(L - 1, !t) / Z
        # if we sum t we get
        # LC(L) = LZ (property 1 of chen 94)
        # log C(l - 1, <t) = lC_forward[..., t - 1, l - 1]
        # log C(L - l, >t) = lC_reverse[..., T - t, counts - l]
        # log w_t = logits[..., t - 1]

        # we need to map
        # lC_reverse'[..., t, l] = lC_reverse[..., T - t - 1, L - l - 1]

        # l -> L - l - 1
        idxs_L = self.condition_count.long()
        idxs_L = idxs_L.unsqueeze(-1).expand(self._batch_shape + (1,))

        idxs_L = torch.where(
            idxs_L > 0,
            (
                idxs_L - torch.arange(1, self._L + 1, device=idxs_L.device)
            ) % idxs_L.clamp(1),
            idxs_L
        )  # handling pytorch modulo 0 crash

        # t -> T - t - 1
        idxs_T = torch.arange(
            self._T - 1, -1, -1, dtype=torch.long, device=idxs_L.device)
        idxs_T = idxs_T.expand(self._batch_shape + self._event_shape)

        idxs = (idxs_T.unsqueeze(-1) * (self._L + 1)) + idxs_L.unsqueeze(-2)
        idxs = idxs.flatten(-2)

        x = self.lC_reverse.flatten(-2).gather(-1, idxs).view(
            *self._batch_shape, self._T, self._L)

        x = x + self.lC_forward[..., :-1, :-1]
        x = x.logsumexp(-1)  # (N, T)
        x = x + self.logits

        x = x.softmax(-1)  # P(t) / L
        return x * self.condition_count.unsqueeze(-1)

    @property
    def variance(self):
        p = self.mean
        return p * (1 - p)

    def sample(self, sample_shape=torch.Size()):
        # the Direct Method of Chen '97
        # remember - we don't have to worry about t past total_count... they
        # have weight zero and will never be picked
        with torch.no_grad():
            logits_shape = self._extended_shape(sample_shape)
            logits = self.logits.expand(logits_shape)
            lC_reverse_shape = (
                logits_shape[:-1] +
                torch.Size((self._T + 1,)) +
                torch.Size((self._L + 1,))
            )

            lC_reverse = self.lC_reverse.expand(lC_reverse_shape)
            n_minus_r = self.condition_count.expand(logits_shape[:-1]).long()
            n_minus_r = n_minus_r.unsqueeze(-1)
            U = (  # noise
                torch.rand_like(logits[..., 0]).log() +
                lC_reverse[..., -1, :].gather(-1, n_minus_r).squeeze(-1)
            )  # (N,)

            b = []
            for t in range(1, self._T + 1):
                # this is numerically unstable - errors accumulate with the
                # subtraction. We should probably switch to the Bounded
                # Bernoulli sampling procedure
                print(n_minus_r[70375, 7], U[70375, 7].exp())
                lC_t = lC_reverse[..., -t - 1, :].gather(
                    -1, n_minus_r).squeeze(-1)
                print(lC_t[70375, 7].exp(), logits[70375, 7, t - 1].exp())
                match = (U >= lC_t) & (n_minus_r.squeeze(-1) > 0)  # hack!
                b.append(match)
                U = torch.where(
                    match,
                    U + torch.log1p(-((lC_t - U).exp())) - logits[..., t - 1],
                    U
                )
                n_minus_r = n_minus_r - match.long().unsqueeze(-1)

            return torch.stack(b, -1).float()
