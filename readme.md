
Implementation of [Does the Data Induce Capacity Control in Deep Learning?](https://arxiv.org/abs/2110.14163) (ICML 2022)




## Set Up
Install backpack package [https://github.com/f-dangel/backpack](https://github.com/f-dangel/backpack)

```
pip3 install backpack-for-pytorch
```

## Usage
Experiments calculating eigenvalues, eigenvectors, overlaps and analytical PAC-Bayes bounds: exp_all.ipynb

Experiments on random datasets: random sloppy.ipynb


Experiments calculating PAC-Bayes bounds: 

Fully-connected net, reproduction of https://arxiv.org/abs/1703.11008 
```
python bayes1.py --num_neurons 600 --num_layers 2
```

Fully-connected net, Method2: 
```
python bayes_kfac1.py --num_neurons 600 --num_layers 2
```

Fully-connected net, Method3: 
```
python bayes_kfac2.py --num_neurons 600 --num_layers 2
```

Fully-connected net, Method4: 
```
python bayes_kfac.py --num_neurons 600 --num_layers 2
```


LENET, reproduction of https://arxiv.org/abs/1703.11008 
```
python bayes2.py 
```

LENET, Method2: 
```
python bayes_proj.py --method method2
```

LENET, Method3: 
```
python bayes_proj.py --method method3
```

LENET, Method4: 
```
python bayes_proj_prior.py 
```
