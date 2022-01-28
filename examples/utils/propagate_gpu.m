function [ outputWavefront ] = propagate_gpu( inputWavefront, dist, ...
    KK, k, method )

% -----------------------------------------------------------------
% This function numerically simulates the propagation of a complex 
% wavefront with a certain method.
% -----------------------------------------------------------------
%   INPUT  [inputWavefront] original complex wavefront
%           [dist] propagating distance (mm)
%           [pxSize] pixel size (mm)
%           [waveLen] wavelength (mm)
%           [method] propagation method ('Fresnel' or 'Angular 
%           Spectrum')
%   OUTPUT  [outputWavefront] complex wavefront after propagation
% -----------------------------------------------------------------

inputFT = fftshift(fft2(inputWavefront));

if strcmp(method,'Fresnel')
    H = exp(1i*k*dist)*exp(-1i*dist*KK/2/k);
elseif strcmp(method,'Angular Spectrum')
    H = exp(1i*dist*sqrt(k^2-KK));
else
    errordlg('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>','Error');
end

outputWavefront = ifft2(fftshift(inputFT.*H));

end

