function [ outputWavefront ] = propagate_gpu( inputWavefront, dist, ...
    KK, kk, method )

inputFT = fftshift(fft2(inputWavefront));

if strcmp(method,'Fresnel')
    H = exp(1i*kk*dist)*exp(-1i*dist*KK/2/kk);
elseif strcmp(method,'Angular Spectrum')
    % remove evanescent orders
    KX_m = KX;
    KY_m = KY;
    ind = (KX.^2+KY.^2 >= kk^2);
    KX_m(ind) = 0;
    KY_m(ind) = 0;
    % transfer function
    H = exp(1i*dist*sqrt(kk^2-KX_m.^2-KY_m.^2));
else
    errordlg('Wrong parameter for [method]: must be <Angular Spectrum> or <Fresnel>','Error');
end

outputWavefront = ifft2(fftshift(inputFT.*H));

end

