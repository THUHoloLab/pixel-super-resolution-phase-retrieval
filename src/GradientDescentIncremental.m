function [x,F_vals,runtimes] = GradientDescentIncremental(x_init,F,dFi,step,n_iters,S,threshold)
x = x_init;
F_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
F_vals(1) = F(x);
eta = 2;
timer = tic;
for iter = 1:n_iters
    for k = 1:S
        x = x - step*dFi(x,k);
    end
    F_vals(iter+1) = F(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, F_vals(iter+1), step, runtimes(iter));
    if (-F_vals(iter+1)+F_vals(iter)) / F_vals(iter) < threshold
        step = step/eta;
    end
end

end

