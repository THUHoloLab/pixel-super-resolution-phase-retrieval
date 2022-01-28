%% generate data
clear;clc;
% close all;

% load functions and test image
addpath(genpath('./utils'))

load('../data/experiment/d1_phase_target.mat')

%% 
% =========================================================================
%                         Data preprocessing
% =========================================================================
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
% =========================================================================
%                         Algorithm initialization
% =========================================================================
addpath(genpath('./utils'))
addpath(genpath('../src'))

%%
sigma = 1;              % down-sampling ratio (along each dimension)
n1 = size(ref,1)*sigma;
n2 = size(ref,2)*sigma;
phasemasks_rs = zeros(n1,n2,S);
for i = 1:S
    phasemasks_rs(:,:,i) = imresize(phasemasks(:,:,i),[n1,n2],'bilinear');
end

% diffraction calculation
kx = pi/(pxsize/sigma)*(-1:2/n2:1-2/n2);
ky = pi/(pxsize/sigma)*(-1:2/n1:1-2/n1);
[KX,KY] = meshgrid(kx,ky);
KK = KX.^2+KY.^2;
k = 2*pi/wavlen;

% measurement operator
A = @(x,j) propagate_gpu(x.*exp(-1i*phasemasks_rs(:,:,j)),dist,KK,k,method);
AH = @(x,j) propagate_gpu(x,-dist,KK,k,method).*exp(1i*phasemasks_rs(:,:,j));

x_init = ones(n1,n2);   % initialization
n_iters = 20;           % number of iteration
step = 2;               % step size

S0 = 64;

myF = @(x) F(x,y,A,S0,sigma);
mydF = @(x) dF(x,y,A,AH,S0,sigma);
mydFi = @(x,k) dFi(x,y,A,AH,sigma,k);

threshold = -Inf;
[x_ggd,F_ggd,~] = GradientDescentGlobal(x_igd,myF,mydF,step,n_iters);

%% display the results
figure,imshow(abs(x_ggd),[])
figure,imshow(phase(x_ggd),[])

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