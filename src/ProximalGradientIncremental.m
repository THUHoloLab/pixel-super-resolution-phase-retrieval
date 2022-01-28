function [x,J_vals,runtimes] = ProximalGradientIncremental(x_init,F,dFi,R,proxR,step,n_iters,S,threshold)
x = x_init;
J_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
J_vals(1) = F(x)+R(x);
eta = 2;
timer = tic;
for iter = 1:n_iters
    for k = 1:S
        x = x - step*dFi(x,k);
    end
    x = proxR(x,step);
    J_vals(iter+1) = F(x)+R(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, J_vals(iter+1), step, runtimes(iter));
    
    if (J_vals(iter) - J_vals(iter+1)) / J_vals(iter) < threshold
        step = step/eta;
    end
end

end

