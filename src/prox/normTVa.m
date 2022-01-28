function norm = normTVa(x)

grad = L(x);
norm = sum(abs(grad),[1,2,3]);

end

