function u = ST(x,sig)
% =========================================================================
% Calculate the transpose of the down-sampling (pixel binning) operator S,
% which corresponds to the nearest interpolation operator.
% -------------------------------------------------------------------------
% Input:    - x   : Low-resolution 2D image.
%           - sig : Down-sampling ratio (positive integer).
% Output:   - u   : Nearest-interpolated 2D image.
% =========================================================================

u = zeros(size(x)*sig);
for r = 0:sig-1
    for c = 0:sig-1
        u(1+r:sig:end,1+c:sig:end) = x;
    end
end

end