# SignSGD_Codec
This project contains research on algorithms that have improved signSGD with majority voting (SignSGD-MV, [paper](https://arxiv.org/abs/1802.04434)) for various problems in distributed learning and federated learning systems. The main purpose of this project is to address data heterogeneity, system heterogeneity (especially computational heterogeneity), and attack-resilience problems based on the excellent communication efficiency of SignSGD-MV. The key idea of algorithm improvement is to model the signSGD-MV learning algorithm as a binary symmetric channel and interpret it from a coding theory perspective.

The list of released papers related to this project is as follows:
* **[Data heterogeneity]** Sparse-SignSGD with majority vote for communication-efficient distributed learning ([arXiv](https://arxiv.org/abs/2302.07475), [ISIT paper](https://ieeexplore.ieee.org/abstract/document/10206480))
* **[System heterogeneity]** SignSGD with federated voting ([arXiv](https://arxiv.org/abs/2403.16372), [ISIT paper](https://ieeexplore.ieee.org/abstract/document/10619155))
* **[Attack-resilience]** SignSGD with federated defense: Harnessing adversarial attacks through gradient sign decoding ([ICML paper](https://proceedings.mlr.press/v235/park24h.html))

## How to implement

### Libraries (with version when uploading code)
* torch: 2.4.0
* numpy: 1.26.4
* matplotlib: 3.8.4 (if you want to plot)
* copy

### Usage
Type the code below:
```
python main.py --[option] [input]
```
If you want to implement this code with basic option, you just type only `python main.py`.

If you want to change the options (e.g., dataset, number of workers, and hyper-parameters), please type `python main.py --help` and check all the options of our code.

If you want to run each algorithm in the above papers, you can apply the following options:
* **[Data heterogeneity]** Sparse-SignSGD with majority voting ($`\mathsf{S}^3`$GD-MV): `python main.py --sparsity 0.1`
* **[System heterogeneity]** SignSGD with federated voting (SignSGD-FV): `python main.py --learning_method FV`
* **[Attack-resilience]** SignSGD with federated defense (SignSGD-FD): `python main.py --learning_method FD`
* **[Baseline]** SignSGD with majority voting (SignSGD-MV): `python main.py --learning_method MV`
