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

%% load experimental data
clear;clc;
close all;

% load functions
addpath(genpath('./utils'))
addpath(genpath('../src'))

group_num = 2;      % group number
load(['../data/experiment/E',num2str(group_num),'.mat'])

%% estimate maximum resolution (optional)
[m1,m2,~] = size(y);
L = min(m1,m2)*pxsize;
sig_max = L*pxsize/(sqrt(dist^2+L^2/4)*wavlen);
disp(['Maximum down-sampling ratio: ',num2str(floor(sig_max)),'x'])

%% crop image to speed up computation (optioinal)
figure
[~,rectAoI2] = imcrop(mat2gray(ref));
rectAoI2 = round(rectAoI2);
if rem(rectAoI2(3),2) == 0       % to ensure the width and length are even numbers
    rectAoI2(3) = rectAoI2(3)-1;
end
if rem(rectAoI2(4),2) == 0
    rectAoI2(4) = rectAoI2(4)-1;
end
ref = imcrop(ref,rectAoI2);  % crop the zero-padded edges

images_crop = zeros([size(ref),K]);
phasemasks_crop = zeros([size(ref),K]);
for k = 1:K
    disp([num2str(k),'/',num2str(K)]);
    images_crop(:,:,k) = imcrop(y(:,:,k),rectAoI2);
    phasemasks_crop(:,:,k) = imcrop(phasemasks(:,:,k),rectAoI2);
end
y = images_crop;
phasemasks = phasemasks_crop;

%%
save(['cache_E',num2str(group_num),'.mat'])

%%
% =========================================================================
% Pixel super-resolution phase retrieval algorithms
% =========================================================================

sig = 1;      % down-sampling ratio (along each dimension)

% zero-pad the object to avoid convolution artifacts
kernelsize = dist*wavlen/(pxsize/sig)/2;    
nullpixels = ceil(kernelsize / (pxsize/sig));
n1 = size(ref,1)*sig + 2*nullpixels;
n2 = size(ref,2)*sig + 2*nullpixels;

phasemasks_rs = zeros(n1,n2,K);
for k = 1:K
    phasemasks_rs(:,:,k) = zeropad(imresize(phasemasks(:,:,k),[size(ref,1)*sig,size(ref,2)*sig],'bicubic'),nullpixels);
end

% pre-calculate diffraction operators
kx = pi/(pxsize/sig)*(-1:2/n2:1-2/n2);
ky = pi/(pxsize/sig)*(-1:2/n1:1-2/n1);
[KX,KY] = meshgrid(kx,ky);
KK = KX.^2+KY.^2;
kk = 2*pi/wavlen;   % wave number

% forward model
Q  = @(x,k) propagate_gpu(x.*exp(-1i*phasemasks_rs(:,:,k)),dist,KK,kk,method);
QH = @(x,k) propagate_gpu(x,-dist,KK,kk,method).*exp(1i*phasemasks_rs(:,:,k));
C  = @(x) imgcrop(x,nullpixels);
CT = @(x) zeropad(x,nullpixels);
A  = @(x,k) C(Q(x,k));
AH = @(x,k) QH(CT(x),k);

x_init = ones(n1,n2);

lam = 1e-4;             % regularization parameter
gam = 2;                % step size

n_iters    = 10;        % number of iterations (main loop)
n_subiters = 1;         % number of iterations (denoising)

% options
opts.verbose = true;
opts.errfunc = nan;
opts.threshold = 1e-3;                                  % threshold for step size update (for incremental algorithms)
opts.eta = 2;                                           % step size decrease ratio (for incremental algorithms)

K = 64;     % total number of images used for reconstruction

% function handles
myF     = @(x) F(x,y,A,K,sig);                          % fidelity function 
mydF    = @(x) dF(x,y,A,AH,K,sig);                      % gradient of the fidelity function
mydFk   = @(x,k) dFk(x,y,A,AH,k,sig);                   % gradient of the fidelity function with respect to the k-th measurement
myR     = @(x) normTV(x,lam);                           % regularization function
myproxR = @(x,gam) proxTV(x,gam,lam,n_subiters);        % proximal operator for the regularization function

% run the algorithm
[x_awf,J_awf,E_awf,runtimes_awf] = AWF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);     % AWF (accelerated Wirtinger flow)
[x_wf, J_wf, E_wf, runtimes_wf ] = WF(x_init,myF,mydF,myR,myproxR,gam,n_iters,opts);      % WF (Wirtinger flow)
[x_wfi,J_wfi,E_wfi,runtimes_wfi] = WFi(x_init,myF,mydFk,myR,myproxR,gam,n_iters,K,opts);  % WFi (Wirtinger flow with incremental updates)

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
x_awf_crop = x_awf(nullpixels+1:end-nullpixels,nullpixels+1:end-nullpixels);
x_wf_crop  = x_wf(nullpixels+1:end-nullpixels, nullpixels+1:end-nullpixels);
x_wfi_crop = x_wfi(nullpixels+1:end-nullpixels,nullpixels+1:end-nullpixels);

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