function w = constraint_simplex_projection(v, z)
% solve probabilistic simplex projection(z = 1), v without any zero elements
n = length(v);

mu = sort(v,'descend'); % descending order.

rho = zeros(n,1);

w = zeros(n,1); 


for j = 1:n
    
    sum = 0;
    for r = 1:j
       sum = sum +  mu(r);
    end
    
    rho(j) = mu(j) -(1.0/j)*(sum-z); 
    
end

% find max j \in [n] where mu_j -1/j(sum_{r=1}^{j} mu_r- z))
rho_max_index = 1;
for j = 1:n
    if(rho(j)>0)
        rho_max_index = j;
    end
end

if(rho_max_index <=0)
   msg = 'Error occurred. max index cannot be zero';
   error(msg)
else
   temp = 0;
   for i = 1:rho_max_index
       temp = temp + mu(i);
   end
   theta = 1/rho_max_index * (temp - z);
end

for i = 1:n
   w(i) =max(v(i) - theta,0);

end