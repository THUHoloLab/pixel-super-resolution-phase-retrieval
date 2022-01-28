function u = D(x,sigma)
    u = zeros(size(x));
    for r = 0:sigma-1
        for c = 0:sigma-1
            u(1:sigma:end,1:sigma:end) = u(1:sigma:end,1:sigma:end) + x(1+r:sigma:end,1+c:sigma:end);
        end
    end
    u = u(1:sigma:end,1:sigma:end);
end
