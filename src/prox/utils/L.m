function u = L(x)

[n1,n2] = size(x);
u = zeros(n1,n2,2);

u(:,:,1) = x - circshift(x,[-1,0]);
u(n1,:,1) = 0;
u(:,:,2) = x - circshift(x,[0,-1]);
u(:,n2,2) = 0;

end

