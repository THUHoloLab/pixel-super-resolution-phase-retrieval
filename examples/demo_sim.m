% ========================================================================
% Introduction
% ========================================================================
% This code provides a simple demonstration of lensless on-chip holographic
% imaging using the pixel super-resolution phase retrieval algorithms.
% 
% For implementation details, please refer to our papers:
% 
%     - Yunhui Gao and Liangcai Cao, "Generalized optimization framework 
%       for pixel super-resolution imaging in digital holography," 
%       Optics Express 29, 28805-28823 (2021). [DOI: 10.1364/OE.434449]
% 
%     - Yunhui Gao, Feng Yang, and Liangcai Cao, "Pixel Super-Resolution 
%       Phase Retrieval for Lensless On-Chip Microscopy via Accelerated 
%       Wirtinger Flow," Cells 11, 1999 (2022). [DOI: 10.3390/cells11131999]
%
% For any further questions, feel free to contact me:
%       Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================

%%
% =========================================================================
% Data generation
% =========================================================================
clear;clc
close all

% load functions
addpath(genpath('./utils'))
addpath(genpath('../src'))

% set parameters
sig = 4;    % down-sampling ratio
m = 256;    % sensor resolution: m x m
n = m*sig;  % sample resolution: n x n
K = 8;      % number of diversity measurements

% load test images
img1 = imresize(im2double(imread('../data/simulation/cameraman.bmp')),[n,n]);
img2 = imresize(im2double(imread('../data/simulation/peppers.bmp')),  [n,n]);

% sample
x = (0.8*img1+0.2).*exp(1i*pi/2*(img2));

% physical parameters
params.pxsize = 5e-3;                   % pixel size (mm)
params.wavlen = 0.5e-3;                 % wavelength (mm)
params.method = 'Angular Spectrum';     % numerical method
params.dist   = 5;                      % imaging distance (mm)

% check model correctness
dist_crit = 2*max([size(x,1),size(x,2)])*params.pxsize^2/params.wavlen;
if dist_crit < max(params.dist)
    error('Angular spectrum not applicable')
end

% zero-pad the object to avoid convolution artifacts
kernelsize = params.dist*params.wavlen/params.pxsize/2; % diffraction kernel size
nullpixels = ceil(kernelsize / params.pxsize);          % number of padding pixels
x = zeropad(x,nullpixels);                              % zero-padded sample

% generate modulation patterns
mask = NaN([size(x),K]);
ptsize = sig*1;     % feature size of the modulation patterns
for k = 1:K
    pattern = imresize(rand(size(x)/ptsize),size(x),'bicubic');
    mask(:,:,k) = exp(1i*pi*pattern);
end

% forward model
Q  = @(x,k) propagate(x.*mask(:,:,k), params.dist,params.pxsize,params.wavlen,params.method);
QH = @(x,k) propagate(x,-params.dist,params.pxsize,params.wavlen,params.method).*conj(mask(:,:,k));
C  = @(x) imgcrop(x,nullpixels);
CT = @(x) zeropad(x,nullpixels);
A  = @(x,k) C(Q(x,k));
AH = @(x,k) QH(CT(x),k);

% generate data
rng(0)              % random seed, for reproducibility
noisevar = 0.01;    % noise level
y = NaN(m,m,K);
for k = 1:K
    u = A(x,k);
    y(:,:,k) = max(S(abs(u).^2,sig).*(1+noisevar*randn(m,m)),0);    % Gaussian noise
end

% display measurement
figure
subplot(1,2,1),imshow(angle(x),[]);colorbar;
title('Phase of the object','interpreter','latex','fontsize',12)
subplot(1,2,2),imshow(y(:,:,1),[]);colorbar;
title('Intensity measurement','interpreter','latex','fontsize',12)
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])

%%
% =========================================================================
% Pixel super-resolution phase retrieval algorithms
% =========================================================================

% define a rectangular region for computing the errors
region.x1 = nullpixels+1;
region.x2 = nullpixels+n;
region.y1 = nullpixels+1;
region.y2 = nullpixels+n;

% algorithm settings
x_init = zeros(size(x));   % initial guess
x_init(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n) = 1;

lam = 1e-3;             % regularization parameter
gam = 2;                % step size (see the paper for details)
n_iters = 50;           % number of iterations (main loop)
n_subiters = 1;         % number of iterations (TV denoising)

% options
opts.verbose = true;                                % display status during the iterations
opts.errfunc = @(z) relative_error_2d(z,x,region);  % user-defined error metrics
opts.threshold = 1e-3;                              % threshold for step size update (for incremental algorithms)
opts.eta = 2;                                       % step size decrease ratio (for incremental algorithms)

% function handles
myF     = @(x) F(x,y,A,K,sig);                          % fidelity function 
mydF    = @(x) dF(x,y,A,AH,K,sig);                      % gradient of the fidelity function
mydFk   = @(x,k) dFk(x,y,A,AH,k,sig);                   % gradient of the fidelity function with respect to the k-th measurement
myR     = @(x) normTV(x,lam);                           % regularization function
myproxR = @(x,gamma) proxTV(x,gamma,lam,n_subiters);    % proximal operator for the regularization function

% run the algorithm
[x_awf,J_awf,E_awf,runtimes_awf] = AWF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);     % AWF (accelerated Wirtinger flow)
[x_wf, J_wf, E_wf, runtimes_wf ] = WF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);      % WF (Wirtinger flow)
[x_wfi,J_wfi,E_wfi,runtimes_wfi] = WFi(x_init,myF,mydFk,myR,myproxR,gam,n_iters,K,opts);  % WFi (Wirtinger flow with incremental updates)

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
x_awf_crop = x_awf(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);
x_wf_crop  = x_wf(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);
x_wfi_crop = x_wfi(nullpixels+1:nullpixels+n,nullpixels+1:nullpixels+n);

% visualize the reconstructed images
figure
subplot(2,3,1),imshow(abs(x_awf_crop),[]);colorbar
title(['Accelerated WF (Obj. Val. = ', num2str(J_awf(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
subplot(2,3,2),imshow(abs(x_wf_crop), []);colorbar
title(['WF (Obj. Val. = ', num2str(J_wf(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
subplot(2,3,3),imshow(abs(x_wfi_crop),[]);colorbar
title(['Incremental WF (Obj. Val. = ', num2str(J_wfi(end),'%4.3f'),')'],'interpreter','latex','fontsize',14)
subplot(2,3,4),imshow(angle(x_awf_crop),[]);colorbar
subplot(2,3,5),imshow(angle(x_wf_crop), []);colorbar
subplot(2,3,6),imshow(angle(x_wfi_crop),[]);colorbar
set(gcf,'unit','normalized','position',[0.15,0.2,0.7,0.6])

figure
semilogy(0:n_iters,J_awf,'linewidth',1.5,'color','r');
hold on,semilogy(0:n_iters,J_wf,'linewidth',1.5,'color','g');
hold on,semilogy(0:n_iters,J_wfi,'linewidth',1.5,'color','b');
legend('AWF','WF','WFi')
%%
% =========================================================================
% Auxiliary functions
% =========================================================================

function v = F(x,y,A,K,sig)
% =========================================================================
% Data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - K   : Total measurement number.
%           - sig : Down-sampling ratio.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
v = 0;
for k = 1:K
    v = v + 1/K*Fk(x,y,A,k,sig);
end

end


function v = Fk(x,y,A,k,sig)
% =========================================================================
% Data-fidelity function w.r.t. the k-th measurement.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - k   : Measurement number of interest.
%           - sig : Down-sampling ratio.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
v = 1/2 * norm2(sqrt(S(abs(A(x,k)).^2,sig)) - sqrt(y(:,:,k)))^2;

function n = norm2(x)   % calculate the l2 vector norm
n = norm(x(:),2);
end

end


function g = dF(x,y,A,AH,K,sig)
% =========================================================================
% Gradient of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
%           - K   : Total measurement number.
%           - sig : Down-sampling ratio.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
g = zeros(size(x));
for k = 1:K
    g = g + 1/K*dFk(x,y,A,AH,k,sig);
end

end


function g = dFk(x,y,A,AH,k,sig)
% =========================================================================
% Gradient of the data-fidelity function w.r.t. the k-th measurement.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity images.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
%           - k   : Measurement number of interest.
%           - sig : Down-sampling ratio.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
u = A(x,k);
a = sqrt(S(abs(u).^2,sig));
g = 1/2 * AH(u.*ST((1./a).*(a - sqrt(y(:,:,k))), sig), k);

end


function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);

end


function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);

end