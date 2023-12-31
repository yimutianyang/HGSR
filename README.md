# TKDE23-HGSR
Implementation of our paper: Hyperbolic Graph Learning for Social Recommendation~(Accepted by TKDE).
The PDF file is available at: https://le-wu.com/files/Publications/JOURNAL/TKDE23-HGSR-yang.pdf
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
* python run_hgsr.py --dataset epinions --negative_sampling random --interest_weight 0.8
* python run_hgsr.py --dataset flickr --negative_sampling pop --interest_weight 0.9
* python run_hgsr.py --dataset ciao --negative_sampling random --interest_weight 0.8
* python run_hgsr.py --dataset dianping --negative_sampling random --interest_weight 0.7

Author contact:
--------------
Email: yyh.hfut@gmail.com

Citation:
--------------
If you find this useful for your research, please kindly cite the following paper:<be>
```
@article{yang2023hyperbolic,
  title={Hyperbolic Graph Learning for Social Recommendation},
  author={Yang, Yonghui and Wu, Le and Zhang, Kun and Hong, Richang and Zhou, Hailin and Zhang, Zhiqiang and Zhou, Jun and Wang, Meng},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
