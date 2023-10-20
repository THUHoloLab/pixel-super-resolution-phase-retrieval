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
% For any further questions, please feel free to contact me:
%       Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================

%%
% =========================================================================
% Data generation
% =========================================================================
clear;clc;
close all;

% load functions
addpath(genpath('./utils'))
addpath(genpath('../src'))

% load experimental data
group_num = 1;  % group number
load(['../data/experiment/E',num2str(group_num),'.mat'])

%% crop image to speed up computation
figure
[~,rectAoI2] = imcrop(mat2gray(ref));
rectAoI2 = round(rectAoI2);
if rem(rectAoI2(3),2) == 0      % to ensure the width and length are even numbers
    rectAoI2(3) = rectAoI2(3)-1;
end
if rem(rectAoI2(4),2) == 0
    rectAoI2(4) = rectAoI2(4)-1;
end
ref = imcrop(ref,rectAoI2);     % crop the zero-padded edges
close

y_crop = zeros([size(ref),K]);
for k = 1:K
    disp([num2str(k),'/',num2str(K)]);
    y_crop(:,:,k) = imcrop(y(:,:,k),rectAoI2);
end
y = y_crop;

% estimate maximum resolution
[m1,m2,~] = size(y);
L = min(m1,m2)*pxsize;
sig_max = L*pxsize/(sqrt(dist^2+L^2/4)*wavlen);
disp(['Maximum down-sampling ratio: ',num2str(floor(sig_max)),'x'])

%%
% =========================================================================
% Pixel super-resolution phase retrieval algorithms
% =========================================================================

gpu = true;

sig = 2;      % down-sampling ratio (along each dimension)

% zero-pad the object to avoid convolution artifacts
nullpixels = 100;
n1 = size(ref,1)*sig + 2*nullpixels*sig;
n2 = size(ref,2)*sig + 2*nullpixels*sig;

% phasemasks_crop = zeros(size(ref,1)+2*nullpixels,size(ref,2)+2*nullpixels,K);
mask = zeros(n1,n2,K);
for k = 1:K
    phasemask_crop = imcrop(phasemasks(:,:,k),...
        [rectAoI2(1)-nullpixels,rectAoI2(2)-nullpixels,rectAoI2(3)+2*nullpixels,rectAoI2(4)+2*nullpixels]);
    mask(:,:,k) = exp(-1i*imresize(phasemask_crop,[size(phasemask_crop,1)*sig,size(phasemask_crop,2)*sig],'bicubic'));
end

% pre-calculate the transfer functions for diffraction modeling
H_f = fftshift(transfunc(n2, n1, dist,pxsize/sig,wavlen,method)); % forward propagation
H_b = fftshift(transfunc(n2, n1,-dist,pxsize/sig,wavlen,method)); % backward propagation

% forward model
M  = @(x,k) x.*mask(:,:,k);             % mask modulation
MH = @(x,k) x.*conj(mask(:,:,k));       % Hermitian of M: conjugate mask modulation
H  = @(x) ifft2(fft2(x).*H_f);          % forward propagation
HH = @(x) ifft2(fft2(x).*H_b);          % Hermitian of H: backward propagation
C  = @(x) imgcrop(x,nullpixels*sig);    % image cropping operation (to model the finite size of the sensor area)
CT = @(x) zeropad(x,nullpixels*sig);    % transpose of C: zero-padding operation
A  = @(x,k) C(H(M(x,k)));               % overall sampling operation
AH = @(x,k) MH(HH(CT(x)),k);            % Hermitian of A

% algorithm settings
x_est = rand(n1,n2);    % initial guess
lam = 1e-3;             % regularization parameter
gam = 2;                % step size
n_iters    = 10;        % number of iterations (main loop)
n_subiters = 1;         % number of iterations (denoising)
K = 64;                 % total number of images used for reconstruction

% auxilary variables
z_est = x_est;
g_est = zeros(size(x_est));
v_est = zeros(size(x_est,1),size(x_est,2),2);
w_est = zeros(size(x_est,1),size(x_est,2),2);

% initialize GPU
if gpu
    device  = gpuDevice();
    x_est   = gpuArray(x_est);
    y       = gpuArray(y);
    mask    = gpuArray(mask);
    H_f     = gpuArray(H_f);
    H_b     = gpuArray(H_b);
    z_est   = gpuArray(z_est);
    g_est   = gpuArray(g_est);
    v_est   = gpuArray(v_est);
    w_est   = gpuArray(w_est);
end

% run WFi to obtain a good initial guess
for iter = 1:1

    % print status
    fprintf('iter: %4d / %4d \n', iter, n_iters);

    % gradient update
    for k = 1:K
        u = A(x_est,k);
        a = sqrt(Sf(abs(u).^2,sig));
        u = 1/2 * AH(u.*STf((1./a).*(a - sqrt(y(:,:,k))), sig), k);
        x_est = x_est - gam * u;
    end

    % proximal update
    v_est(:) = 0; w_est(:) = 0;
    for subiter = 1:n_subiters
        w_next = v_est + 1/8/gam*Df(x_est-gam*DTf(v_est));
        w_next = min(abs(w_next),lam).*exp(1i*angle(w_next));
        v_est = w_next + subiter/(subiter+3)*(w_next-w_est);
        w_est = w_next;
    end
    x_est = x_est - gam*DTf(w_est);

end

z_est = x_est;

% main loop
timer = tic;
for iter = 1:n_iters

    % print status
    fprintf('iter: %4d / %4d \n', iter, n_iters);
    
    % gradient update
    g_est(:) = 0;
    for k = 1:K
        u = A(z_est,k);
        a = sqrt(Sf(abs(u).^2,sig));
        u = 1/2 * AH(u.*STf((1./a).*(a - sqrt(y(:,:,k))), sig), k);
        g_est = g_est + 1/K*u;
    end
    u = z_est - gam * g_est;

    % proximal update
    v_est(:) = 0; w_est(:) = 0;
    for subiter = 1:n_subiters
        w_next = v_est + 1/8/gam*Df(u-gam*DTf(v_est));
        w_next = min(abs(w_next),lam).*exp(1i*angle(w_next));
        v_est = w_next + subiter/(subiter+3)*(w_next-w_est);
        w_est = w_next;
    end
    x_next = u - gam*DTf(w_est);
    
    % Nesterov extrapolation
    z_est = x_next + (iter/(iter+3))*(x_next - x_est);
    x_est = x_next;
end

% wait for GPU
if gpu; wait(device); end
toc(timer)

% gather data from GPU
if gpu
    x_est   = gather(x_est);
    y       = gather(y);
    H_f     = gather(H_f);
    H_b     = gather(H_b);
    mask    = gather(mask);
    reset(device);
end

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
x_est_crop = C(x_est);

% visualize the reconstructed images
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
subplot(1,2,1),imshow(abs(x_est_crop),[]);colorbar
title('Retrieved amplitude','interpreter','latex','fontsize',14)
subplot(1,2,2),imshow(angle(x_est_crop),[]);colorbar
title('Retrieved phase','interpreter','latex','fontsize',14)

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


function H = transfunc(nx, ny, dist, pxsize, wavlen, method)
% =========================================================================
% Calculate the transfer function of the free-space diffraction.
% -------------------------------------------------------------------------
% Input:    - nx, ny   : The image dimensions.
%           - dist     : Propagation distance.
%           - pxsize   : Pixel (sampling) size.
%           - wavlen   : Wavelength of the light.
%           - method   : Numerical method ('Fresnel' or 'Angular Spectrum').
% Output:   - H        : Transfer function.
% =========================================================================

% sampling in the spatial frequency domain
kx = pi/pxsize*(-1:2/nx:1-2/nx);
ky = pi/pxsize*(-1:2/ny:1-2/ny);
[KX,KY] = meshgrid(kx,ky);

k = 2*pi/wavlen;    % wave number

ind = (KX.^2 + KY.^2 >= k^2);  % remove evanescent orders
KX(ind) = 0; KY(ind) = 0;

if strcmp(method,'Fresnel')
    H = exp(1i*k*dist)*exp(-1i*dist*(KX.^2+KY.^2)/2/k);
elseif strcmp(method,'Angular Spectrum')
    H = exp(1i*dist*sqrt(k^2-KX.^2-KY.^2));
else
    errordlg('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>','Error');
end
end


function w = Df(x)
% =========================================================================
% Calculate the 2D gradient (finite difference) of an input image.
% -------------------------------------------------------------------------
% Input:    - x  : The input 2D image.
% Output:   - w  : The gradient (3D array).
% =========================================================================
w = cat(3,x(1:end,:) - x([2:end,end],:),x(:,1:end) - x(:,[2:end,end]));
end


function u = DTf(w)
% =========================================================================
% Calculate the transpose of the gradient operator.
% -------------------------------------------------------------------------
% Input:    - w  : 3D array.
% Output:   - x  : 2D array.
% =========================================================================
u1 = w(:,:,1) - w([end,1:end-1],:,1);
u1(1,:) = w(1,:,1);
u1(end,:) = -w(end-1,:,1);

u2 = w(:,:,2) - w(:,[end,1:end-1],2);
u2(:,1) = w(:,1,2);
u2(:,end) = -w(:,end-1,2);

u = u1 + u2;
end


function u = Sf(x,sig)
% =========================================================================
% Calculate the down-sampling (pixel binning) operator.
% -------------------------------------------------------------------------
% Input:    - x   : High-resolution 2D image.
%           - sig : Down-sampling ratio (positive integer).
% Output:   - u   : Down-sampled 2D image.
% =========================================================================

u = x;
u(:) = 0;
for r = 0:sig-1
    for c = 0:sig-1
        u(1:sig:end,1:sig:end) = u(1:sig:end,1:sig:end) + x(1+r:sig:end,1+c:sig:end);
    end
end
u = u(1:sig:end,1:sig:end);

end


function u = STf(x,sig)
% =========================================================================
% Calculate the transpose of the down-sampling (pixel binning) operator S,
% which corresponds to the nearest interpolation operator.
% -------------------------------------------------------------------------
% Input:    - x   : Low-resolution 2D image.
%           - sig : Down-sampling ratio (positive integer).
% Output:   - u   : Nearest-interpolated 2D image.
% =========================================================================

u = imresize(x,size(x)*sig,"nearest");

end