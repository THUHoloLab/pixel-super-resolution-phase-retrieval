function dx = dF(x,y,A)

S = size(y,3);
sigma = round(size(x,1)/size(y,1));

dx = 0;
for k = 1:S
    u = A(x,k);
    a = sqrt(D(abs(u).^2,sigma));
    e = a - sqrt(y(:,:,k));
    dx = dx + 1/2/S*norm2(e);
end

end

