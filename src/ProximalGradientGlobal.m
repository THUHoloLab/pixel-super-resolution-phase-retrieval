function [x,J_vals,runtimes] = ProximalGradientGlobal(x_init,F,dF,R,proxR,step,n_iters)
x = x_init;
J_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
J_vals(1) = F(x)+R(x);
timer = tic;
for iter = 1:n_iters
    x = x - step*dF(x);
    x = proxR(x,step);
    J_vals(iter+1) = F(x)+R(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, J_vals(iter+1), step, runtimes(iter));
end

end

