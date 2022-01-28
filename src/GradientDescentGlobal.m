function [x,F_vals,runtimes] = GradientDescentGlobal(x_init,F,dF,step,n_iters)
x = x_init;
F_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
F_vals(1) = F(x);
timer = tic;
for iter = 1:n_iters
    x = x - step*dF(x);
    F_vals(iter+1) = F(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, F_vals(iter+1), step, runtimes(iter));
end
end

