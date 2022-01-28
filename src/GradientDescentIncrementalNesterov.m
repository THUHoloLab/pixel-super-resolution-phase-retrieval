function [x,F_vals,runtimes] = GradientDescentIncrementalNesterov(x_init,F,dFi,step,n_iters,S,threshold)
x = x_init;
y = x;
a = 1;
F_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
F_vals(1) = F(x);
eta = 2;
timer = tic;
for iter = 1:n_iters
    a_next = (1 + sqrt(1 + 4*a^2))/2;
    for k = 1:S
        x_next = y - step*dFi(y,k);
        y = x_next + (a-1)/a_next*(x_next - x);
        x = x_next;
    end
    a = a_next;
    F_vals(iter+1) = F(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, F_vals(iter+1), step, runtimes(iter));
    if (-F_vals(iter+1)+F_vals(iter)) / F_vals(iter) < threshold
        step = step/eta;
        threshold = threshold;
    end
end
end

