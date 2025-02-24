function X = dense_sylvester(B,C)
    
    [n1,n2] = size(B);
    N = n1*n2;
    b = reshape(B,N,1);
    A = zeros(N,N);
    nmat=size(C,1);
    for i = 1:nmat
        A = A + kron(C{i,2}',C{i,1});
    end
    x = A\b;
    X = reshape(x,n1,n2);
end

