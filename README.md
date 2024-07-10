# Collective Certified Robustness against Graph Injection Attacks
 This is the official source code of the paper "Collective Certified Robustness against Graph Injection Attacks" (ICML2024).

Collective certification against graph injection attack in the node classification task.

### Environment setup

```bash
cd ./Environments
conda env create -f py37.yml
conda activate py37
pip install -r py37.txt
```
or to specific dir:
```bash
cd ./Environments
conda env create -f py37.yml -p /home/xxx/py37
conda activate /home/xxx/py37
pip install -r py37.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yml or .txt file and delete that line.


### Data Preparation
./Data: it contains the Cora-ML and Citeseer datasets.  
./results_citeseer and ./results_cora: They contain the smoothing sampling and their predicted results. It is required for the optimization problem.
It can be prepared from the source code of the sample-wise certificate: https://github.com/Yuni-Lai/NodeAwareSmoothing. Please refer to the paper for details:
```angular2html
Lai, Yuni, et al. "Node-aware Bi-smoothing: Certified Robustness against Graph Injection Attacks." IEEE Symposium on Security and Privacy (2023).
```

### Collective certification
#### Run with bash for all parameters and settings:
```bash
nohup bash run.sh > ./run.log 2>&1 &
```

#### Run with single parameter:
```bash
python main.py -dataset 'cora' -p_e 0.8 -p_n 0.9 -optimization 'LP2'
```

### Citation
If you think this repo is helpful, please include the citation as follows:
```bash
@inproceedings{
lai2024collective,
title={Collective Certified Robustness against Graph Injection Attacks},
author={Lai, Yuni and Pan, Bailin and Chen, Kaihuang and Yuan, Yancheng and Zhou, Kai},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}

```


