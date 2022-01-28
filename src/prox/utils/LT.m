function div = LT(u)

[n1,n2,~] = size(u);

shift = circshift(u(:,:,1),[1,0]);
div1 = u(:,:,1) - shift;
div1(1,:) = u(1,:,1);
div1(n1,:) = -shift(n1,:);

shift = circshift(u(:,:,2),[0,1]);
div2 = u(:,:,2) - shift;
div2(:,1) = u(:,1,2);
div2(:,n2) = -shift(:,n2);

div = div1 + div2;

end

