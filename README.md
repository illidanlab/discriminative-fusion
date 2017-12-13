# Discriminative fusion of multiple brain networks

## Introduction

In neuroimaging research, brain networks derived from different tractography methods may lead to different results and perform differently when used in classification tasks. As there is no ground truth to determine which brain network models are most accurate or most sensitive to group differences, we developed a new sparse learning method that combines information from multiple network models. We used it to learn a convex combination of brain connectivity matrices from multiple different tractography methods, to optimally distinguish people with early mild cognitive  impairment from  healthy control subjects, based on the structural connectivity patterns. 

## Usage
The main function is called classify.m. The inputs of this function is as follows: 

`result_dir`: directory to save results

`task_name`: name of the task, e.g. 'classification AD with MCI'

`X`:input data. It is a n*d*m tensor. n is sample size, d is feature dimension, m is modalities number.

`group`: labels of all samples, i.e., 1, 2, 3

`pos_class_arr`: postive class labels array. For example, if `group == 1` and `group == 2` all denote postive class, `pos_class_arr = [1, 2]` 

`neg_class_arr`: negative class lables array.

`repeat_exp`: number of validaton folds. 

`method_str`: 'fusion'

`classifier_param`: parameter to control model complexity to prevent overfitting

`overwrite`: if True, the current setting will overwrite existing results

## Citation

As you use this code for your exciting discoveries, please cite the paper below:

> Wang, Qi, Liang Zhan, Paul M. Thompson, Hiroko H. Dodge, and Jiayu Zhou. "Discriminative fusion of multiple brain networks for early mild cognitive impairment detection." In Biomedical Imaging (ISBI), 2016 IEEE 13th International Symposium on, pp. 568-572. IEEE, 2016.

Or if you use Bibtex:

```
@inproceedings{wang2016discriminative,
  title={Discriminative fusion of multiple brain networks for early mild cognitive impairment detection},
  author={Wang, Qi and Zhan, Liang and Thompson, Paul M and Dodge, Hiroko H and Zhou, Jiayu},
  booktitle={Biomedical Imaging (ISBI), 2016 IEEE 13th International Symposium on},
  pages={568--572},
  year={2016},
  organization={IEEE}
}
```

The paper can be downloaded [here](http://jiayuzhou.github.io/papers/qwangISBI16.pdf). 
