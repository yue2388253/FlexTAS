# FlexTAS

A deep reinforcement learning-based scheduler for TSN scheduling.

This repo also contains Tabu and TimeTabling scheduler.

# How to use

* Run tests

```shell
PYTHONPATH=. python -m unittest
```

* Train the model

```shell
PYTHONPATH=. python app/train.py --time_steps 1_000_000 --jitters 0.1 
```

* Run the experiments

```shell
PYTHONPATH=. python src/app/evaluation.py \
    --topos RRG,ERG,BAG --list_num_flows 10,60,110,160,210,260,310 \
    --link_rate 100 --num_tests 100 --list_obj drl --jitters 0.1,0.2,0.5 \
    --seed 2345 --num_non_tsn_devices 2 --drl_model model/best_model.zip \
    --to_csv out/test.csv --timeout 5
```

Plz check the codes for more available options.

# Prerequisite

We provide two `yml` files for create conda env, one is for CPU-only machine, and the other is for GPU-enabled machine.

```shell
conda env create -f conda_env.yml
```

## Known issues

Issue 1. After installing conda env, it may still have problems.

How to fix: reinstall the package using pip. e.g.: 

```shell
pip install --force-reinstall numpy
```

---
