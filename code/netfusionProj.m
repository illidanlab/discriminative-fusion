function [model, output] = netfusionProj(X_ndm, y, rho, options )
% constrained version. 

if nargin < 4
    options = pnopt_optimset(...
        'debug'         , 0      ,... % debug mode
        'desc_param'    , 0.0001 ,... % sufficient descent parameter
        'display'       , 0    ,... % display frequency (<= 0 for no display)
        'backtrack_mem' , 10     ,... % number of previous function values to save
        'max_fun_evals' , 50000  ,... % max number of function evaluations
        'max_iter'      , 1000   ,... % max number of iterations
        'ftol'          , 1e-9   ,... % stopping tolerance on objective function
        'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
        'xtol'          , 1e-9    ... % stopping tolerance on solution
        );
end


n = size(X_ndm, 1);
d = size(X_ndm, 2);
m = size(X_ndm, 3);

x0         = randn(d + 1 + m, 1);  % init point
smoothF    = @(x) fusion_loss (X_ndm, y, x);
nonsmoothF = prox_segment_l1(rho,  1:d, d + 2:d + 1 + m);

[ x, ~, output ] = pnopt_sparsa( smoothF, nonsmoothF, x0, options );

w   = reshape(x(1:d), [d, 1]);
c   = x(d+1);
tau = x(end-m + 1 : end);

fprintf('NNZ for w:   %u\n', nnz(w));
fprintf('NNS for tau: %u\n', nnz(tau));

model.w   = w;
model.c   = c;
model.tau = tau;
% a function for prediction.
model.predict     = @(X_dm)  sign((w' * X_dm * tau) + c);
model.predict_set = @(X_ndm) sign(reshape(w' * reshape(permute(X_ndm, [2, 3, 1]), d, m* n), m, n)' * tau + c);
end

function [f_val, grad] = fusion_loss (X_ndm, y, model_vect)
% Solves the function value and gradient of the loss function.
% Fusion loss:
%   f(X, y) = sum_i log(1+exp(p_i)) + ||w||_1
%      where p_i = -y_i ( sum_{m=1}^M tau_m (X_i^(m)^T w) +c )
%
%  X:   n X d X m
%  y:   n X 1
%
%  [w; m; d]
%    tau: m X 1
%    w:   d X 1
%

n = size(X_ndm, 1);
d = size(X_ndm, 2);
m = size(X_ndm, 3);

w   = reshape(model_vect(1:d), [d, 1]);
c   = model_vect(d+1);
tau = model_vect(end-m + 1 : end);

X_dmn = permute(X_ndm, [2, 3, 1]);

% possible weighting
weight = ones(n, 1)/n;
weighty = weight.* y;


% compute p_i
wX_mn = reshape(w' * reshape(X_dmn, d, m* n), m, n);
aa = -y.* ( wX_mn' * tau + c);
bb = max (aa, 0);
f_val = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );

pp = 1./ (1+exp(aa));

b = -weighty.*(1-pp);

grad_w   = reshape(reshape(X_ndm, n* d, m) * tau, n, d)' * b;
grad_c   = sum(b);
grad_tau = wX_mn * b;

grad = [grad_w(:); grad_c; grad_tau(:)];

end


function op = prox_segment_l1( q, segment_l1, segment_simplex )

%PROX_L1    L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).

if nargin == 0,
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0)
    error( 'Argument must be positive.' );
end

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes

    function v = f(x)
        v = norm( q.*x(segment_l1), 1 );
    end

    function x = prox_f(x,t)
        % project l1 part. 
        tq = t .* q; 
        s  = 1 - min( tq./abs(x(segment_l1)), 1 );
        x(segment_l1) = x(segment_l1) .* s;
        
        % project the simplex part. 
        x(segment_simplex) = ...
            constraint_simplex_projection(x(segment_simplex), 1);
    end


end
