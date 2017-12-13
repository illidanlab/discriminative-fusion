function [prediction, perf_info] = apply_classify_netfusion(w, c,tau, X_test, Y_test)
n = size(X_test, 1);
d = size(X_test, 2);
m = size(X_test, 3);
confidence = reshape(w' * reshape(permute(X_test, [2, 3, 1]), d, m* n), m, n)' * tau + c;
%(X_ndm) sign(reshape(w' * reshape(permute(X_ndm, [2, 3, 1]), d, m* n), m, n)' * tau + c)


prediction = sign(confidence);
 

    info                  = perfStat(Y_test, prediction);
    perf_info.auc         = computeAUC(Y_test, confidence); 
    perf_info.accuracy    = info.accuracy;
    perf_info.f1          = info.f1;
    perf_info.sensitivity = info.sensitivity;
    perf_info.specificity = info.specificity;
end