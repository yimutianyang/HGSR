# TKDE23-HGSR
Implementation of our paper: Hyperbolic Graph Learning for Social Recommendation~(Accepted by TKDE)
![](https://github.com/yimutianyang/HGSR/blob/main/framework.jpg)

In this work, we investigate hyperbolic graph learning for social recommendation. 
We argue that two challenges limit hyperbolic learning to social recommendation: heterogeneity and social noise introduced by explicit diffusion.
Therefore, we design a social pre-training enhanced hyperbolic heterogeneous graph learning method, named HGSR.
Specifically, we first pre-train social networks in hyperbolic space, which can preserve the hierarchical structure properties. 
Next, we feed the pre-trained social embeddings into a hyperbolic heterogeneous graph for preference learning. 
Such that, we combine explicit heterogeneous graph learning implicit social feature enhancement for hyperbolic social recommendation, 
which can effectively tackle heterogeneity and noise issues. Experiments on four datasets show the effectiveness of the proposed model, 
including high performance, generalization of the pre-trained feature, and applicability to various sparsity users.

Prerequisites
-------------
* Please refer requirements.txt

Usage
-----
* python run_hgsr.py --dataset epinions --runid 3layer+random_sample --num_layers 3 --negative_sampling random --interest_weight 0.8
* python run_hgsr.py --dataset flickr --runid 3layer+pop_sample --num_layers 3 --negative_sampling pop --interest_weight 0.9

Author contact:
--------------
Email: yyh.hfut@gmail.com
