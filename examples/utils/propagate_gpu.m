function [ outputWavefront ] = propagate_gpu( inputWavefront, dist, ...
    KK, kk, method )



inputFT = fftshift(fft2(inputWavefront));

if strcmp(method,'Fresnel')
    H = exp(1i*kk*dist)*exp(-1i*dist*KK/2/kk);
elseif strcmp(method,'Angular Spectrum')
    H = exp(1i*dist*sqrt(kk^2-KK));
else
    errordlg('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>','Error');
end

outputWavefront = ifft2(fftshift(inputFT.*H));

end

