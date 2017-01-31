rand('state',0)
n=50;
p=n;
w=sprandsym(n,p);
%w=(w>0)-(w<0); % this choice defines a frustrated system
w=(w>0); % this choice defines a ferro-magnetic (easy) system
w=w-diag(diag(w));


w = zeros(n, n);