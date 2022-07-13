# Conditional Bernoulli

## What is this?

This codebase was developed for the paper "Computing a Likelihood over Latent
Binary Variables with a Fixed Sum." It should be considered development code
not suited for general usage. I plan on eventually moving this functionality to
one of my public repos with better documentation. For now, consult the paper on
what everything is.

## Installation

This is not a proper Python package. Commands are expected to be run directly
from the directory this file is contained in. A Conda Python environment called
`conditional-bernoulli` with the necessary dependencies (and unnecessary ones;
anything used anywhere in this repo) can be installed with the command

``` sh
conda create -f environment.yml
```

though you can easily crawl the `environment.yml` file for the package names to
install via PyPI.

## Running

### TIMIT

The TIMIT recipe can be found in `timit.sh`. Running

``` sh
./timit.sh -i /path/to/timit
```

will train a model for every valid combination of `<model>_<estimator>_<lm>`
and a fixed number of seeds *in sequence*.

`<model>` can be one of

- `indep`: both the latent and conditional models obey the Markov Assumption.
- `partial`: the latent model obeys the Markov Assumption, not the conditional.
- `full`: neither the latent nor conditional obeys the Markov Assumption.

`<estimator>` can be one of

- `direct`: A naive Monte Carlo estimate which draws samples directly from the
  latent distribution (without any conditioning).
- `marginal`: Marginalizes out the latent distribution. Only possible with the
  `indep` model.
- `cb`: A Monte Carlo estimate which draws from the latent distribution
  conditioned on the number of events. Not possible for the `full` model.
- `srswor`: A Monte Carlo Importance Sampling estimate with a proposal
  uniformly distributed over event configurations given the number of events
  (Simple Random Sampling WithOut Replacement).
- `ais-c`: The AIS-IMH algorithm with sufficient statistics estimated by count.
- `ais-g`: The AIS-IMH algorithm with sufficient statistics estimated by Gibbs
  conditionals.

`<lm>` refers to part of the conditional model which predicts the next token
based on the history of tokens, like a traditional LM. AM predictions are
injected via weighted mixture. It can be one of

- `lm-flatstart`: Sequence prediction is trained alongside everything else.
- `lm-pretrained`: Sequence prediction is trained first. Its weights are frozen
  while the rest of the model is trained.
- `nolm`: Sequence prediction is minimal


## Testing

``` sh
pytest *.py
```

Tests are at the end of each Python file. Don't judge me.

## Licensing

If this repository is public on GitHub, it is Apache 2.0-licensed. If you're
looking at an anonymous version for review purposes, please don't distribute.