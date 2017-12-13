function [  ] = test_grad(  )
% Test the gradient of function value

n = 202;
m = 9;
d = 500;


X_ndm = randn(n, d, m);
y     = randn(n, 1);


vect0 = randn(m + d + 1, 1) * 10;


test_func = @(x)fusion_loss (X_ndm, y, x);

check_grad(test_func, vect0);



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

function check_grad(f, x0, varargin)
% a simple function that checks the correctness of gradient. 
% INPUT
%  f  - a function handle of f(x) that returns function values and gradients given parameter x 
%  x0 - the location near which the gradient will be evaluted. 
%
% For a correct gradiet, the displayed ratio should be near 1.0  

delta = rand(size(x0));
delta = delta ./ norm(delta);
epsilon = 10.^(-7:-1);

[~, df0] = feval(f, x0, varargin{:});

for i = 1:length(epsilon)
    [f_left] = feval(f, x0-epsilon(i)*delta, varargin{:});
    [f_right] = feval(f, x0+epsilon(i)*delta, varargin{:});
    ys(i) = (f_right - f_left) / 2;      
    ys_hat(i) = df0' * epsilon(i)*delta;    
    fprintf('epsilon: %d , gradient: %d \n', epsilon(i), ys(i) / ys_hat(i));
end         

end