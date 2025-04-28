# Forecasting Company Fundamentals

Official implementation of the paper *Forecasting Company Fundamentals* submitted to TMLR.

## Installation

```shell
pip install -r requirements.txt
pip install -e .
```

## Data

The data is not included in this repository due legal restrictions. Unfortunately, it is not publicly available.

## Overview

Some supporting code is provided in `forecasting_cfs`, mamely helers for evaulations and models not yet available in the main `darts` distribution.
The main experimenation is provided in `notebboks/`, which includes the following notebooks:
- `run.ipynb`: Produce the forecasts for the paper. Also provides explanations of the TFT model.
- `eval_results.ipynb`: Analyize the models trained in `run.ipynb`.
- `compare_with_human_eval.ipynb`: Produce the comparisons with human evaluations.
- `combine_for_market_evaluation.ipynb`: Collect and condense the results for the realistic market evaluation.
