function [x,F_vals,runtimes] = GradientDescentGlobalNesterov(x_init,F,dF,step,n_iters)
x = x_init;
y = x;
a = 1;
F_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
F_vals(1) = F(x);
timer = tic;
for iter = 1:n_iters
    x_next = y - step*dF(y);
    F_vals(iter+1) = F(x_next);
    a_next = (1 + sqrt(1 + 4*a^2))/2;
    y = x_next + (a-1)/a_next*(x_next - x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, F_vals(iter+1), step, runtimes(iter));
    a = a_next;
    x = x_next;
end
end

