function lowip =inner_low(U,S,V,Ut,St,Vt)
% return the inner product of two low rank matrices
% safety check, USVT, must be same dim as the other matrix

r1=size(S,1);
r2=size(St,1);

lowip = 0;
for i=1:r1
    utranspose = U(:,i)';
    vtranspose = V(:,i)';
    sloc = S(i,i);
    for j=1:r2
        lowip=lowip + sloc*St(j,j)*utranspose*Ut(:,j)*vtranspose*Vt(:,j);
    end
end

end