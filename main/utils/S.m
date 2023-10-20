function u = S(x,sig)
% =========================================================================
% Calculate the down-sampling (pixel binning) operator.
% -------------------------------------------------------------------------
% Input:    - x   : High-resolution 2D image.
%           - sig : Down-sampling ratio (positive integer).
% Output:   - u   : Down-sampled 2D image.
% =========================================================================

u = zeros(size(x));
for r = 0:sig-1
    for c = 0:sig-1
        u(1:sig:end,1:sig:end) = u(1:sig:end,1:sig:end) + x(1+r:sig:end,1+c:sig:end);
    end
end
u = u(1:sig:end,1:sig:end);

end
