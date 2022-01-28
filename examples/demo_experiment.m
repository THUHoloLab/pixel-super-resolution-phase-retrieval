%% generate data
clear;clc;
% close all;

% load functions and test image
addpath(genpath('./utils'))

load('../data/experiment/d1_phase_target.mat')

%% estimate maximum resolution (optional)
[m1,m2,~] = size(y);
L = min(m1,m2)*pxsize;
maxScale = L*pxsize/(sqrt(dist^2+L^2/4)*wavlen);
disp(['Maximum upsampling ratio: ',num2str(floor(maxScale)),'x'])

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

images_crop = zeros([size(ref),S]);
phasemasks_crop = zeros([size(ref),S]);
for i = 1:S
    disp([num2str(i),'/',num2str(S)]);
    images_crop(:,:,i) = imcrop(y(:,:,i),rectAoI2);
    phasemasks_crop(:,:,i) = imcrop(phasemasks(:,:,i),rectAoI2);
end
y = images_crop;
phasemasks = phasemasks_crop;

%%
save('data_cell_v0.mat')

%%
clear all
load('data_cell_v0.mat')

addpath(genpath('./utils'))
addpath(genpath('../src'))

%%
sigma = 1;      % down-sampling ratio (along each dimension)
n1 = size(ref,1)*sigma;
n2 = size(ref,2)*sigma;
phasemasks_rs = zeros(n1,n2,S);
for i = 1:S
    phasemasks_rs(:,:,i) = imresize(phasemasks(:,:,i),[n1,n2],'bilinear');
end

kx = pi/(pxsize/sigma)*(-1:2/n2:1-2/n2);
ky = pi/(pxsize/sigma)*(-1:2/n1:1-2/n1);
[KX,KY] = meshgrid(kx,ky);
KK = KX.^2+KY.^2;
k = 2*pi/wavlen;

A = @(x,j) propagate_gpu(x.*exp(-1i*phasemasks_rs(:,:,j)),dist,KK,k,method);
AH = @(x,j) propagate_gpu(x,-dist,KK,k,method).*exp(1i*phasemasks_rs(:,:,j));

n_iters = 20;
x_init = ones(n1,n2);
step = 2;

S0 = 64;

myF = @(x) F(x,y,A,S0,sigma);
mydF = @(x) dF(x,y,A,AH,S0,sigma);
mydFi = @(x,k) dFi(x,y,A,AH,sigma,k);

threshold = -Inf;
[x_igd,F_igd,~] = GradientDescentIncremental(x_init,myF,mydFi,step,1,S,threshold);
[x_aggd,F_aggd,~] = GradientDescentGlobal(x_igd,myF,mydF,step,n_iters);

% lambda = 5e-3;
% n_subiters = 2;
% mydF2 = @(x) mydF(x)*2;
% myR = @(x) lambda*normTVa(x);    % isotropic TV norm as the penalty function
% myproxR = @(x,gamma) proxTVa(x,gamma*lambda,n_subiters);       % proximity operator
% [x_agpg,F_agpg,runtimes_agpg] = ProximalGradientGlobalNesterov(x_igd,myF,mydF2,myR,myproxR,1,n_iters);

%%
figure,imshow(abs(flipud(x_aggd)),[])
%%
figure,imshow(abs(flipud(x_agpg)),[0,1])
%%
x_new = propagate_gpu(x_aggd,0.013,KK,k,method);
figure,imshow(abs(fliplr(x_new)),[])
%%
figure,imshow(angle(x),[-pi+1,pi]);colorbar
figure,plot(0:10,mean(objs,2))
figure,plot(0:50,[mean(objs,2);mean(objs2(2:end,:),2)]);
%%
save('results/cell_FOV.mat','x_aggd','F_aggd')

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