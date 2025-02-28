function [U_OFT,U_EX]=DOFT2(N);

    xmin = -1.0;
    xmax =  1.0;
    dx = (xmax-xmin) / (N+1);
    x  = (xmin + dx:dx:xmax - dx)';
    hx = dx;
    hy = dx;
    y = x;
    U_ex = exp(-36 * x.^2);
    V_ex = exp(-36 * y.^2);
    S_ex = norm(U_ex, 2) * norm(V_ex, 2);
    U_ex = U_ex ./ norm(U_ex, 2);
    V_ex = V_ex ./ norm(V_ex, 2);

    % Do a direct solve
    e = ones(N,1);
    A_direct = spdiags([e -2*e e], -1:1, N, N);
    f = U_ex * S_ex * V_ex';
    f_vec = f(:);
    L = kron(A_direct, speye(N, N)) + kron(speye(N, N), A_direct);
    L = speye(N * N, N * N) - (1i/dx^2) * L;
    U_OFT = L \ f_vec;
    U_OFT = reshape(U_OFT, N, N);
    U_EX = U_ex * S_ex * V_ex';
end