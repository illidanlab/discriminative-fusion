function [perf_struct] = perfStat (Y, Y_pred, opts)
pos_class_lb = +1;
neg_class_lb = -1;

if (nargin<3)
    opts = [];
end

if isfield(opts, 'pos_class_lb')
    pos_class_lb = opts.pos_class_lb;
end
if isfield(opts, 'neg_class_lb')
    neg_class_lb = opts.neg_class_lb;
end

tp = nnz(Y == Y_pred & Y_pred == pos_class_lb);
fp = nnz(Y ~= Y_pred & Y_pred == pos_class_lb);
tn = nnz(Y == Y_pred & Y_pred == neg_class_lb);
fn = nnz(Y ~= Y_pred & Y_pred == neg_class_lb);
total = tp+fp+tn+fn;

accuracy  = (tp + tn)/total;
if tp + fp ~= 0
    precision = tp / (tp + fp);
else
    precision = 1;
end
if tp + fn ~= 0
    recall    = tp / (tp + fn);
else
    recall    = 1;
end
sensitivity = recall;
if tn+fp ~=0
    specificity = tn / (tn + fp);
else
    specificity = 1;
end

if (precision + recall) == 0
    f1 = 0;
else
    f1 = 2 * precision * recall / (precision + recall);
end

if sensitivity + specificity == 0
    h1 = 0;
else
    h1 = 2 * sensitivity * specificity / (sensitivity + specificity);
end

perf_struct.accuracy = accuracy;
perf_struct.precision = precision;
perf_struct.recall = recall;
perf_struct.specificity = specificity;
perf_struct.sensitivity = sensitivity;
perf_struct.f1 = f1;
perf_struct.h1 = h1;


end