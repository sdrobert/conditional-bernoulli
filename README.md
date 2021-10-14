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
`conditional-bernoulli` with the necessary dependencies (and unnecessary ones,
anything used anywhere in this repo) can be installed with the command

``` sh
conda create -f environment.yml
```

though you can easily crawl the `environment.yml` file for the package names to
install via PyPI.

## Running

If you are interested in just the descriptive statistics of the experiment (as
well as how I generated the tables and figures), open up the Jupyter notebook
`df_vis.ipynb`. Results per iteration are stored in `res/df.*.csv.gz` as
gzipped CSV files if you want to look at the data directly.

`df_bernoulli.py` contains the command-line interface for training a single
Drezner & Farnum statistical model to match an objective.

``` sh
python df_bernoulli.py --print-ini > df.ini  # store default config
# modify df.ini with your own parameters
python df_bernoulli.py --read-ini df.ini out.csv  # store results in out.csv
```

If you want to reproduce the results found in the `res/` directory, have a look
at `build_array.sh`. I used this script to build a SLURM array of jobs with
different configuration files over a fixed number of seeds, then stored the
gzipped csvs back in to the `res/` folder. You'll have to modify
`build_array.sh` with your own cluster configuration. Then

``` sh
./build_array.sh
sbatch ./df.sh  # runs all of them
sbatch --array=1-3  # runs the first three configurations (see conf/df.*.ini)
```

and sit back for a few days.

## Testing

``` sh
pytest *.py
```

Tests are at the end of each Python file. Don't judge me.

## Licensing

If this repository is public on GitHub, it is Apache 2.0-licensed. If you're
looking at an anonymous version for review purposes, please don't distribute.