function [x,n_iters,objs,errs,runtimes] = WirtingerPSR(A,AH,y,sigma,varargin)
% *************************************************************************
% * This function applies the Wirtinger gradient descent algorithm to solve
%   pixel-super-resolved phase retrieval problems of the form
%
%                 y_k = D | A_k x |^2,    k = 1,2,...,S,
%
%   where x is the complex-valued object transmittance, y_k is the kth
%   image, A_k is the forward transmittance matrix for the kth measurement,
%   D is the downsampling matrix, and S is the measurement number.
%
%   See the following paper for details.
%
%   [1] Yunhui Gao and Liangcai Cao, "Generalized optimization framework 
%       for pixel super-resolution imaging in digital holography," 
%       Opt. Express 29, 28805-28823 (2021)
%
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- A  : function handle
%          The forward operator of the system. 
%
%   - AH : function handle
%          The Hermitian of A.
%
%	- y  : 3D array of shape (m,n,S)
%          The intensity measurements. m and n are the width and height 
%          of the intensity images, and S is the measurement number.
%
% 	- sigma : int
%             Upsampling ratio. Should be a positive integer.
%
%   ===== Optional inputs =================================================
%
%   - 'initializer'  : must be one of {0, 1, 2, 3, 4, 2D array}, default = 0
%                      Initializer for x:
%                      0 -> uniform amplitude & zero phase
%                      1 -> uniform amplitude & random phase
%                      2 -> random amplitude & uniform phase
%                      3 -> random amplitude & random phase
%                      4 -> averaged back-propagated amplitude & zero phase
%                      array -> initialization provided by the user
%
%   - 'max_iter' : int, default = 100
%                  Maximum number of iterations allowed.
%
%   - 'min_iter' : int, default = 0
%                  Minimum number of iterations allowed.
%        
%   - 'stop_criterion' : must be one of {0, 1, 2, 3}, default = 0
%                        Type of stopping criterion to use:
%                        0 -> stop when the relative change in the
%                             objective function falls below 'tol',
%                        1 -> stop when the relative norm of the  
%                             difference between two consecutive  
%                             estimates falls below 'tol',
%                        2 -> stop when the objective function 
%                             becomes equal or less than 'tol',
%                        3 -> stop only when the number of iteration
%                             reaches 'max_iter'.
%
%   - 'tol' : float, default = 1e-8
%             Tolerance, used for stopping criterion.
%
%   - 'step_size' : float, default = 2
%                   Step size for the gradient descent update.
%
%   - 'obj_func'  : must be one of {'amplitude', 'intensity'}
%                   The objective function.
%
%   - 'update_mode' : must be one of {'incremental', 'global'}, default = 'incremental'
%                     The objective function.
%
%   - 'verbose'  : bool, default = false
%                  Visualizes the iterative process.
%
%   - 'ground_truth'  : 2D array, default: None
%                       The ground truth of x.
%
%   - 'err_func'      : function handle, default = @RE
%                       Error function used for comparison with the ground 
%                       truth. The default function RE calculates the 
%                       relative error.
%
%   - 'phase_unwrapper' : function handle, default: None
%                         Function handle that implements phase unwrapping
%                         before calculating the errors.
%
%   ===== Outputs =========================================================
%
%   - x        : 2D array
%                The solution.
%
%   - n_iters  : int
%                Number of iterations.
%
%   - objs     : 1D array
%                The values of the objective function for each iteration.
%
%   - errs     : 1D array
%                The values of the error function for each iteration.
%
%   - runtimes : 1D array
%                Runtime of the algorithm for each iteration.
%
% *************************************************************************
%%

max_iter = 10;
min_iter = 0;
stop_criterion = 0;
tol = 1e-8;
step = 2;
verbose = true;
init = 0;
[m,n,S] = size(y);
obj_func = 'amplitude';
update_mode = 'incremental';
x_gt = NaN;
err_func = @RE;
unwrapper = [];

if (nargin-length(varargin)) ~= 4
    error('Wrong number of required inputs');
elseif rem(length(varargin),2) == 1
    error('Optional inputs should always go by pairs')
end
for i = 1:2:length(varargin)-1
    switch lower(varargin{i})
        case 'initializer'
            if numel(varargin{i+1}) > 1   % we have an initial x
            	init = 5;
            	x = varargin{i+1};
            else
                init = varargin{i+1};
                if (sum(init == [0 1 2 3 4]) == 0)
                    error('Unknwon initializer (''initializer'')');
                end
            end
        case 'max_iter'
            max_iter = varargin{i+1};
        case 'min_iter'
            min_iter = varargin{i+1};
        case 'stop_criterion'
            stop_criterion = varargin{i+1};
        case 'tol'
            tol = varargin{i+1};
        case 'step_size'
            step = varargin{i+1};
        case 'obj_func'
            obj_func = varargin{i+1};
        case 'verbose'
            verbose = varargin{i+1};
        case 'update_mode'
            update_mode = varargin{i+1};
        case 'ground_truth'
            x_gt = varargin{i+1};
        case 'err_func'
            err_func = varargin{i+1};
        case 'phase_unwrapper'
            unwrapper = varargin{i+1};
        otherwise
            error(['Invalid parameter: ',varargin{i}]);
    end
end

switch init
    case 0
        x = ones(m*sigma,n*sigma);
    case 1
        x = exp(1i*2*pi*rand(m*sigma,n*sigma));
    case 2
        x = rand(m*sigma,n*sigma);
    case 3
        x = rand(m*sigma,n*sigma).*exp(1i*2*pi*rand(m*sigma,n*sigma));
    case 4
        x = zeros(m*sigma,n*sigma,S);
        for k = 1:S
            x(:,:,k) = AH(DT(y(:,:,k))/sigma^2,k);
        end
        x = mean(abs(x),3);
    case 5   % initial x was given as a function argument; just check size
        if size(x) ~= size(AH(DT(y(:,:,1)),1))
            error('Size of initial x is not compatible with A'); 
        end
    otherwise
        error('Unknown initialization option (''initializer'')');
end

%% 
% =========================================================================
%                         auxilary functions
% =========================================================================

% The downsampling operator D
function xd = D(x)
    xd = zeros(size(x));
    for r = 0:sigma-1
        for c = 0:sigma-1
            xd(1:sigma:end,1:sigma:end) = xd(1:sigma:end,1:sigma:end) ...
                + x(1+r:sigma:end,1+c:sigma:end);
        end
    end
    xd = xd(1:sigma:end,1:sigma:end);
end

% The adjoint operator of D
function xu = DT(x)
    xu = zeros(size(x)*sigma);
    for r = 0:sigma-1
        for c = 0:sigma-1
            xu(1+r:sigma:end,1+c:sigma:end) = x;
        end
    end
end

% Calculate l2 norm for multi-dimensional arrays
function val = norm2(x)
    val = sqrt(dot(x(:),x(:)));
end

% Relative error (RE) as default error function
function val = RE(x)
    if ishandle(unwrapper)
        pha = unwrapper(angle(x));
    else
        pha = angle(x);
    end
    amp = abs(x);
    pha_norm = pha - mean(mean(pha)) + mean(mean(angle(x_gt)));
    x_norm = amp.*exp(1i*pha_norm);
    val = norm2(x_norm-x_gt)/norm2(x_norm);
end

%% 
% =========================================================================
%                               main loop
% =========================================================================
objs = zeros(max_iter+1,S);
objs(1,:) = Inf(S,1);
errs = [];
if ~isnan(x_gt)
    errs = NaN(max_iter+1,1);
    errs(1) = err_func(x);
end
runtimes = NaN(max_iter,1);
iter = 1;
crit = Inf;
loop = true;
timer = tic;
while loop
    x_next = x;
    grad = NaN([size(x),S]);
    if strcmp(lower(obj_func),'amplitude')
        for k = 1:S
            u = A(x_next,k);
            a = sqrt(D(abs(u).^2));
            e = a - sqrt(y(:,:,k));
            grad(:,:,k) = 1/2*AH(u.*DT((1./a).*e),k);
            objs(iter+1,k) = 1/2/S*norm2(e);
            if strcmp(lower(update_mode),'incremental')
                x_next = x_next - step*grad(:,:,k);             
            end
        end
    elseif strcmp(lower(obj_func),'intensity')
        for k = 1:S
            u = A(x_next,k);
            e = D(abs(u).^2) - y(:,:,k);
            grad(:,:,k) = 1/2*AH(u.*DT(e),k);
            objs(iter+1,k) = 1/2/S*norm2(e);
            if strcmp(lower(update_mode),'incremental')
                x_next = x_next - step*grad(:,:,k);             
            end
        end
    end
    if strcmp(lower(update_mode),'global')
        x_next = x_next - step*mean(grad,3);
    end
    runtimes(iter) = toc(timer);
    
    % check the stopping criterion
    switch stop_criterion
        case 0
            crit = abs(sum(objs(iter+1,:))-sum(objs(iter,:)))/sum(objs(iter,:));
        case 1
            crit = norm2(x_next-x)/norm2(x);
        case 2
            crit = sum(objs(iter+1,:));
    end
    
    if iter >= min_iter
        if iter >= max_iter || crit < tol
            loop = false;
        end
    end
    
    if verbose
        fprintf(['iter: %4d | objective: %10.4e | stepsize: %2.2e | ', ...
            'criterion: %10.4e | runtime: %5.1f s\n'], ...
            iter, sum(objs(iter+1,:)), step, crit, runtimes(iter));
    end
    
    if ~isnan(x_gt)
        errs(iter+1,:) = err_func(x_next);
    end
    iter = iter + 1;
    x = x_next;
    
end

% display result
if verbose
    fprintf('Algorithm terminated.\n')
end

runtimes(isnan(runtimes)) = [];
objs(isnan(runtimes)) = [];
n_iters = iter - 1;

end
