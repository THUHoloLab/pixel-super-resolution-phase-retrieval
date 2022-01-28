%% generate data
clear;clc;
% close all;

% load functions and test image
addpath(genpath('../src'))
addpath(genpath('./utils'))

M = 64;         % pixel number
sigma = 2;      % up-sampling ratio
N = M*sigma/2;    % subpixel number
N = M*sigma;

img1 = im2double(imresize(imread('../data/chart.tif'),[N,N]));
img2 = im2double(imresize(imread('../data/testpat.tif'),[N,N]));
x = (1 - img1*0.5).*exp(1i*pi*(1 - img2));
% x = (1 - img1*0.5);
% x = exp(1i*pi*(1-img2));

img1 = im2double(imresize(imread('../data/peppers.tif'),[N,N]));
img2 = im2double(imresize(imread('../data/cameraman.tif'),[N,N]));
x = (0.8*img1+0.2).*exp(1i*pi/2*(img2));

% x = padarray(x,[N/2,N/2]);

S = 8;     % number of measurements

params.pxsize = 5e-3;      % pixel size (mm)
params.wavlen = 5e-4;      % wavelength (mm)
params.method = 'Angular Spectrum';
params.dist = linspace(1,5,S);   % imaging distance (multi-distance phase retrieval)

dist_crit = 2*max([size(x,1),size(x,2)])*params.pxsize^2/params.wavlen;
if dist_crit < max(params.dist)
    error('Angular spectrum not applicable')
end

% function handles for forward and backward propagators
A = @(x,k) propagate(x, params.dist(k), params.pxsize, params.wavlen, params.method);
AH = @(x,k) propagate(x, -params.dist(k), params.pxsize, params.wavlen, params.method);

% rng(0)
y = NaN(M,M,S);
noise = 0e-3;
for k = 1:S
    u = A(x,k);
    y(:,:,k) = D(abs(u).^2,sigma);
    y(:,:,k) = y(:,:,k).*(1 + noise*randn(M,M));   % add some noise
    y(:,:,k) = max(y(:,:,k) + noise.*randn(M,M),0);
end

%%
[m,n,~] = size(y);
x_init = zeros(m*sigma,n*sigma,S);
for k = 1:S
    x_init(:,:,k) = AH(sqrt(DT(y(:,:,k),sigma)/sigma^2),k);
end
x_init = mean(x_init,3);
% x_init = zeros(size(x_init));

myF = @(x) F(x,y,A,S,sigma);
mydF = @(x) dF(x,y,A,AH,S,sigma);
mydFi = @(x,k) dFi(x,y,A,AH,sigma,k);

n_iters = 400;
step = 2;

%%
[x_aggd,F_aggd,runtimes_aggd] = GradientDescentGlobalNesterov(x_init,myF,mydF,step,n_iters);

%%
[x_ggd,F_ggd,runtimes_ggd] = GradientDescentGlobal(x_init,myF,mydF,step,n_iters);

%%
threshold = 1e-4;
[x_igd,F_igd,runtimes_igd] = GradientDescentIncremental(x_init,myF,mydFi,step,n_iters,S,threshold);

%%
threshold = 1e-4;
[x_aigd,F_aigd,runtimes_aigd] = GradientDescentIncrementalNesterov(x_init,myF,mydFi,step,n_iters,S,threshold);

%%
bias = 0.0;
figure
semilogy([0:n_iters],F_aggd-bias,'linewidth',1)
hold on,semilogy([0:n_iters],F_ggd-bias,'linewidth',1)
hold on,semilogy([0:n_iters],F_igd-bias,'linewidth',1)
hold on,semilogy([0:n_iters],F_aigd-bias,'linewidth',1)
legend('A-GGD','GGD','IGD','A-IGD')

%%
%{
mydF2 = @(x) mydF(x)*2;
lambda = 1e-4;
n_subiters = 1;    % number of iterations to solve the denoising subproblem
myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
[x_gpg,~,F_gpg,runtimes_gpg] = FISTA(myF,mydF2,1,x_init,...        % run FISTA
'prox_op',      myproxR,...
'penalty',      myR,...
'eta',          2,...
'Lip',          1,... 
'max_iter',     n_iters,...
'min_iter',     n_iters,...
'monotone',     true,...
'verbose',      true);
%}

%%
lambda = 5e-4;
n_subiters = 5;    % number of iterations to solve the denoising subproblem

%%
mydF2 = @(x) mydF(x)*2;
myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
[x_gpg,F_gpg,runtimes_gpg] = ProximalGradientGlobal(x_init,myF,mydF2,myR,myproxR,1,n_iters);

%%

mydF2 = @(x) mydF(x)*2;
myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
[x_agpg,F_agpg,runtimes_agpg] = ProximalGradientGlobalNesterov(x_init,myF,mydF2,myR,myproxR,1,n_iters);

%%
mydFi2 = @(x,k) mydFi(x,k)*2;
myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
[x_ipg,F_ipg,runtimes_ipd] = ProximalGradientIncremental(x_init,myF,mydFi2,myR,myproxR,1,n_iters,S);

%%
mydFi2 = @(x,k) mydFi(x,k)*2;
myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
% myR = @(x) 0;
% myproxR = @(x,gamma) x;
[x_aipg,F_aipg,runtimes_aipd] = ProximalGradientIncrementalNesterov(x_init,myF,mydFi2,myR,myproxR,1/5,n_iters,S);

%%
figure
semilogy([0:n_iters],F_gpg)
hold on,semilogy([0:n_iters],F_ipg)
hold on,semilogy([0:n_iters],F_agpg)
hold on,semilogy([0:n_iters],F_aipg)
legend('Global','Incremental','Global accelerated','Incremental accelerated')

%%
function val = F(x,y,A,S,sigma)
val = 0;
for k = 1:S
    val = val + 1/2/S*norm2(sqrt(D(abs(A(x,k)).^2,sigma)) - sqrt(y(:,:,k)))^2;
end
end

function dx = dF(x,y,A,AH,S,sigma)
dx = zeros(size(x));
for k = 1:S
    u = A(x,k);
    a = sqrt(D(abs(u).^2,sigma));
    e = a - sqrt(y(:,:,k));
    dx = dx + 1/2/S*AH(u.*DT((1./a).*e,sigma),k);
end
end

function dx = dFi(x,y,A,AH,sigma,k)
u = A(x,k);
a = sqrt(D(abs(u).^2,sigma));
e = a - sqrt(y(:,:,k));
dx = 1/2*AH(u.*DT((1./a).*e,sigma),k);
end

% calculate the 2-norm of a vector
function val = norm2(x)
    val = sqrt(dot(x(:),x(:)));
end