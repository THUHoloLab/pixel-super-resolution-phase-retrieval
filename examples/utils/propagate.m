function [ wavefront_out ] = propagate( wavefront_in, dist, ...
    px_size, wavelen, method )

% -----------------------------------------------------------------
% This function numerically simulates the propagation of a complex 
% wavefront with a certain method.
% -----------------------------------------------------------------
%   INPUT   [inputWavefront] original complex wavefront
%           [dist] propagating distance (mm)
%           [pxSize] pixel size (mm)
%           [waveLen] wavelength (mm)
%           [method] propagation method ('Fresnel' or 'Angular 
%           Spectrum')
%   OUTPUT  [outputWavefront] complex wavefront after propagation
% -----------------------------------------------------------------

[N,M] = size(wavefront_in);    % size of the wavefront

kx = pi/px_size*(-1:2/M:1-2/M);
ky = pi/px_size*(-1:2/N:1-2/N);
[KX,KY] = meshgrid(kx,ky);

k = 2*pi/wavelen;   % wave number

inputFT = fftshift(fft2(wavefront_in));

if strcmp(method,'Fresnel')
    H = exp(1i*k*dist)*exp(-1i*dist*(KX.^2+KY.^2)/2/k);
elseif strcmp(method,'Angular Spectrum')
    H = exp(1i*dist*sqrt(k^2-KX.^2-KY.^2));
else
    errordlg('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>','Error');
end

wavefront_out = ifft2(fftshift(inputFT.*H));

end

