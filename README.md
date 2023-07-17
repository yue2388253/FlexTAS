# DeepFTAS

灵活门控，全量调度，No-wait

# action

选择哪一条流进行调度，参考论文 
Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning

# Prerequisite

z3

```shell
cd /path/to/your/project
git clone https://https://github.com/Z3Prover/z3.git
cd z3
mkdir build
cd build
cmake ..
make 
sudo make install
```

Some other packages (might lack some packages and need to be resolved manually.)

```shell
# assume an Ubuntu OS
sudo apt install graphviz
```

Python-related packages are managed by `conda`.

```shell
conda env create -f conda_env.yml
```