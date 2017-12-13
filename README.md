# Introduction

In neuroimaging research, brain networks derived from different tractography methods may lead to different results and perform differently when used in classification tasks. As there is no ground truth to determine which brain network models are most accurate or most sensitive to group differences, we developed a new sparse learning method that combines information from multiple network models. We used it to learn a convex combination of brain connectivity matrices from multiple different tractography methods, to optimally distinguish people with early mild cognitive  impairment from  healthy control subjects, based on the structural connectivity patterns. 

# Usage
The main function is called classify.m. The inputs of this function is as follows: 

result_dir: directory to save results

task_name: name of the task, e.g. 'classification AD with MCI'

X:input data. It is a n*d*m tensor. n is sample size, d is feature dimension, m is modalities number.

group: labels of all samples, i.e., 1, 2, 3

pos_class_arr: postive class labels array. For example, if group == 1 and group == 2 all denote postive class, pos_class_arr = [1, 2] 

neg_class_arr: negative class lables array.

repeat_exp: number of validaton folds. 

method_str: 'fusion'

classifier_param: parameter to control model complexity to prevent overfitting

overwrite: if True, the current setting will overwrite existing results



