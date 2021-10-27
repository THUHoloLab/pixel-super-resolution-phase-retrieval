function xd = D(x,sigma)
    xd = zeros(size(x));
    for r = 0:sigma-1
        for c = 0:sigma-1
            xd(1:sigma:end,1:sigma:end) = xd(1:sigma:end,1:sigma:end) + x(1+r:sigma:end,1+c:sigma:end);
        end
    end
    xd = xd(1:sigma:end,1:sigma:end);
end
