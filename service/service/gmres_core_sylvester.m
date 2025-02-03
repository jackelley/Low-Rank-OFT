function [X,FLAG,RELRES,ITER] = gmres_core_sylvester(B,C, tol, itermax)
    
    [n1,n2] = size(B);
    % this is nx, ny
    n1 = size(C{1,1},1);
    n2 = size(C{1,2},1);
    N = n1*n2;
    b = reshape(B,N,1);
    [x,FLAG,RELRES,ITER] = gmres(@(x)get_Ax(x,C),b,min(40,N-1),tol,itermax,[],[],b);
    X = reshape(x,n1,n2);
end

function Ax = get_Ax(x,C)
    nmat=size(C,1);
    n1 = size(C{1,1},1);
    n2 = size(C{1,2},1);
    
    X = reshape(x,n1,n2);
    Ax = zeros(n1,n2);
    for i = 1:nmat
        Ax = Ax + C{i,1}*X*C{i,2};
    end
    Ax = reshape(Ax,n1*n2,1);
end
