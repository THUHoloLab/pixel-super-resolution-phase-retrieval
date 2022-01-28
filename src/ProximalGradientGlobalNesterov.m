function [x,J_vals,runtimes] = ProximalGradientGlobalNesterov(x_init,F,dF,R,proxR,step,n_iters)
x = x_init;
y = x;
a = 1;
J_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
J_vals(1) = F(x)+R(x);
timer = tic;
for iter = 1:n_iters
    x_next = proxR(y - step*dF(y),step);
    J_vals(iter+1) = F(x)+R(x);
    a_next = (1 + sqrt(1 + 4*a^2))/2;
    y = x_next + (a-1)/a_next*(x_next - x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, J_vals(iter+1), step, runtimes(iter));
    a = a_next;
    x = x_next;
end

end

