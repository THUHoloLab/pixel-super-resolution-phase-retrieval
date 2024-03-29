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

% forward model
M  = @(x,k) x.*mask(:,:,k);                             % mask modulation
MH = @(x,k) x.*conj(mask(:,:,k));                       % Hermitian of M: conjugate mask modulation
H  = @(x) propagate(x, dist,pxsize/sig,wavlen,method);  % forward propagation
HH = @(x) propagate(x,-dist,pxsize/sig,wavlen,method);  % Hermitian of H: backward propagation
C  = @(x) imgcrop(x,nullpixels*sig);                    % image cropping operation (to model the finite size of the sensor area)
CT = @(x) zeropad(x,nullpixels*sig);                    % transpose of C: zero-padding operation
A  = @(x,k) C(H(M(x,k)));                               % overall sampling operation
AH = @(x,k) MH(HH(CT(x)),k);                            % Hermitian of A

% algorithm settings
x_init = rand(n1,n2);   % initial guess
lam = 1e-4;             % regularization parameter
gam = 2;                % step size
n_iters    = 10;        % number of iterations (main loop)
n_subiters = 1;         % number of iterations (denoising)
K = 64;                 % total number of images used for reconstruction

% options
opts.verbose = true;
opts.errfunc = nan;
opts.threshold = 1e-3;                                  % threshold for step size update (for incremental algorithms)
opts.eta = 2;                                           % step size decrease ratio (for incremental algorithms)

% function handles
myF     = @(x) F(x,y,A,K,sig);                          % fidelity function 
mydF    = @(x) dF(x,y,A,AH,K,sig);                      % gradient of the fidelity function
mydFk   = @(x,k) dFk(x,y,A,AH,k,sig);                   % gradient of the fidelity function with respect to the k-th measurement
myR     = @(x) normTV(x,lam);                           % regularization function
myproxR = @(x,gam) proxTV(x,gam,lam,n_subiters);        % proximal operator for the regularization function

% run the algorithm
[x_wfi,J_wfi,E_wfi,runtimes_wfi] = WFi(x_init,myF,mydFk,myR,myproxR,gam,1,K,opts);        % WFi for a good initial guess
[x_awf,J_awf,E_awf,runtimes_awf] = AWF(x_wfi, myF,mydF,myR,myproxR,gam,n_iters,opts);      % AWF (accelerated Wirtinger flow)

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
x_awf_crop = C(x_awf);

% visualize the reconstructed images
figure
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])
subplot(1,2,1),imshow(abs(x_awf_crop),[]);colorbar
title('Retrieved amplitude','interpreter','latex','fontsize',14)
subplot(1,2,2),imshow(angle(x_awf_crop),[]);colorbar
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