function x = proxTV(v,gam,lam,n_iters)
% =========================================================================
% Calculate the proximal operator for the TV function.
% -------------------------------------------------------------------------
% Input:    - v       : 2D image.
%           - gam     : The step size.
%           - lam     : The regularization weight.
%           - n_iters : The iteration number.
% Output:   - x       : 2D array.
% =========================================================================

[n1,n2] = size(v);
w = zeros(n1,n2,2);
w_prev = zeros(n1,n2,2);
z = zeros(n1,n2,2);

for t = 1:n_iters
    w = z + 1/8/gam*D(v-gam*DT(z));
    w = min(abs(w),lam).*exp(1i*angle(w));
    z = w + t/(t+3)*(w-w_prev);
    w_prev = w;
end

x = v - gam*DT(w);

end

