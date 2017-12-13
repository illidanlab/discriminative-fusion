clear; clc;


    
n = 202;
m = 9;
d = 500;
X_ndm = randn(n, d, m);
y     = sign(randn(n, 1));

options = pnopt_optimset(...
    'debug'         , 0      ,... % debug mode
    'desc_param'    , 0.0001 ,... % sufficient descent parameter
    'display'       , 100    ,... % display frequency (<= 0 for no display)
    'backtrack_mem' , 10     ,... % number of previous function values to save
    'max_fun_evals' , 50000  ,... % max number of function evaluations
    'max_iter'      , 1000   ,... % max number of iterations
    'ftol'          , 1e-9   ,... % stopping tolerance on objective function
    'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
    'xtol'          , 1e-9    ... % stopping tolerance on solution
    );


sparsity_arr = [0.000001, 0.001, 0.01, 0.05];

for sparsity = sparsity_arr

    [model] = netfusionProj(X_ndm, y, sparsity, options);
    fprintf('sparsity at %.4f is %u\n', sparsity , nnz(model.w));
    fprintf('simplex sum: %.4f\n', sum(model.tau))
    
    % compute the training prediction.
    acc = sum(sign(model.predict_set(X_ndm).* y)>0)/size(X_ndm, 1);
    fprintf('accuracy %.4f percent\n', acc);
    
end