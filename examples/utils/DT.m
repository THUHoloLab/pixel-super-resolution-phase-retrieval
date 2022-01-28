function u = DT(x,sigma)
    u = zeros(size(x)*sigma);
    for r = 0:sigma-1
        for c = 0:sigma-1
            u(1+r:sigma:end,1+c:sigma:end) = x;
        end
    end
end