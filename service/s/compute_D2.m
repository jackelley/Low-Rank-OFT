function D2u = compute_D2(U,n,r,D,BSt,IP,Eb,sig)

% Compute the second derivative with penalty term included
    D2u = zeros(n,r);
    D2u = D*U;
    D2u = D2u + (IP*BSt*Eb-sig*IP*Eb)*U;
end
