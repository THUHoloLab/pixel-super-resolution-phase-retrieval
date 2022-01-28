function [x,J_vals,runtimes] = ProximalGradientIncrementalNesterov(x_init,F,dFi,R,proxR,step,n_iters,S)
x = x_init;
y = x;
a = 1;
J_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);
J_vals(1) = F(x)+R(x);
threshold = 1e-3;
eta = 1.5;
timer = tic;
for iter = 1:n_iters
    a_next = (1 + sqrt(1 + 4*a^2))/2;
    for k = 1:S
        x_next = y - step*dFi(y,k);
        y = x_next + (a-1)/a_next*(x_next - x);
        x = x_next;
    end
    x_next = proxR(y,step);
    y = x_next + (a-1)/a_next*(x_next - x);
    x = x_next;
    a = a_next;
    J_vals(iter+1) = F(x)+R(x);
    runtimes(iter) = toc(timer);
    fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n'], ...
            iter, J_vals(iter+1), step, runtimes(iter));
    
    if (J_vals(iter) - J_vals(iter+1)) / J_vals(iter) < threshold
        step = step/eta;
        threshold = threshold/eta;
    end
end

end

