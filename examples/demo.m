%% generate data
clear;clc;
% close all;

% load functions and test image
addpath(genpath('../src'))
addpath(genpath('./utils'))

M = 64;         % pixel number
sigma = 4;      % up-sampling ratio
N = M*sigma;    % subpixel number

img1 = im2double(imresize(imread('../data/chart.tif'),[N,N]));
img2 = im2double(imresize(imread('../data/testpat.tif'),[N,N]));
x = (1 - img1*0.5).*exp(1i*pi*(1 - img2));
% x = (1 - img1*0.5);
% x = exp(1i*pi*(1-img2));

img1 = im2double(imresize(imread('../data/peppers.tif'),[N,N]));
img2 = im2double(imresize(imread('../data/cameraman.tif'),[N,N]));
x = (0.5 + img1*0.5).*exp(1i*pi*(img2));

pxsize = 1e-3;      % pixel size (mm)
wavlen = 5e-4;      % wavelength (mm)
method = 'Angular Spectrum';

%% generate measurement data (multi-distance)
S = 64;     % number of measurements
d = linspace(1,5,S);   % imaging distance (multi-distance phase retrieval)
y = NaN(M,M,S);
for k = 1:S
    u = propagate(x, d(k), pxsize, wavlen, method);
    y(:,:,k) = D(abs(u).^2,sigma);
    y(:,:,k) = y(:,:,k).*(1 + 5e-3*randn(M,M));   % add some noise
end

% function handles for forward and backward propagators
A = @(x,k) propagate(x, d(k), pxsize, wavlen, method);
AH = @(x,k) propagate(x, -d(k), pxsize, wavlen, method);

%% generate measurement data (coded aperture)
S = 4;     % number of measurements
d = 5;
mask = zeros(N,N,S);
y = NaN(M,M,S);
for k = 1:S
    mask(:,:,k) = exp(1i*imresize(rand(M,M),[N,N]));
    u = propagate(x.*mask(:,:,k), d, pxsize, wavlen, method);
    y(:,:,k) = D(abs(u).^2,sigma);
    y(:,:,k) = y(:,:,k).*(1 + 0e-3*randn(M,M));   % add some noise
end

% function handles for forward and backward propagators
A = @(x,k) propagate(x.*mask(:,:,k), d, pxsize, wavlen, method);
AH = @(x,k) propagate(x, -d, pxsize, wavlen, method).*conj(mask(:,:,k));

%% reconstruction
n_iters = 200;
step = 2;
unwrapper = @(x) puma_ho(x,1);
[x_r,n_iters,objs,errs,~] = WirtingerPSR(A,AH,y,sigma,...
    'initializer',0,...
    'max_iter',n_iters,...
    'min_iter',n_iters,...
    'update_mode','incremental',...
    'step_size',step,...
    'ground_truth',x,...
    'obj_func','amplitude',...
    'phase_unwrapper',unwrapper);

phase = unwrapper(angle(x_r));  % phase unwrapping

figure
subplot(1,2,1),imshow(abs(x_r),[])
subplot(1,2,2),imshow(phase,[])

figure
subplot(1,2,1),plot(0:n_iters,errs)

%%
figure,imshow(abs(x_r)-abs(x),[]);colorbar
colormap jet
figure,imshow(phase-angle(x)-mean(phase-angle(x)),[]);colorbar
colormap jet
