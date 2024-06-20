# FlexTAS

A Deep Reinforcement Learning-based scheduler for TSN. 
It does not enable gating control for all streams. Instead, only a subset of streams are enabled gating. 
The agent learns to whether to enable gating for the given stream and the node.

Note: This repo also contains other schedulers such as Tabu[1], and Oliver2018[2] (The implementation is partially inspired by [OpenPlanner](https://gitee.com/opentsn/open-planner)).

## References:

[1] Frank Dürr and Naresh Ganesh Nayak. 2016. No-wait Packet Scheduling for IEEE Time-sensitive Networks (TSN). In Proceedings of the 24th International Conference on Real-Time Networks and Systems (RTNS '16). Association for Computing Machinery, New York, NY, USA, 203–212. https://doi.org/10.1145/2997465.2997494

[2] R. Serna Oliver, S. S. Craciunas and W. Steiner, "IEEE 802.1Qbv Gate Control List Synthesis Using Array Theory Encoding," 2018 IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), Porto, Portugal, 2018, pp. 13-24, doi: 10.1109/RTAS.2018.00008. keywords: {Logic gates;Schedules;Indexes;Microsoft Windows;Switches;Real-time systems;Synchronization;tsn;scheduling;networks;smt},

# Prerequisite

We provide two `yml` files for create conda env, one is for CPU-only machine, and the other is for GPU-enabled machine.

```shell
conda env create -f conda_env.yml
```

We recommend using a CPU-only machine for training the DRL agent, as the task primarily demands CPU resources rather than GPU capabilities.

# How to use

* Run tests

```shell
PYTHONPATH=. python -m unittest
```

* Train your model

```shell
PYTHONPATH=. python app/train.py --time_steps 1_000_000 --jitters 0.1 
```

* Run evaluations

```shell
PYTHONPATH=. python src/app/evaluation.py \
    --topos RRG,ERG,BAG --list_num_flows 10,60,110,160,210,260,310 \
    --link_rate 100 --num_tests 100 --list_obj drl --jitters 0.1,0.2,0.5 \
    --seed 2345 --num_non_tsn_devices 2 --drl_model model/best_model.zip \
    --to_csv out/test.csv --timeout 5
```

Plz check the codes for more available options.

## Known issues

Issue 1. After installing conda env, it may still have problems.

How to fix: reinstall the package using pip. e.g.: 

```shell
pip install --force-reinstall numpy
```

---
