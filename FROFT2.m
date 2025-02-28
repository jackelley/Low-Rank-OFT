function [U_OFT] = FROFT2(N,dt,Tf);
    addpath('service');
    xmin = -1.0;
    xmax =  1.0;
    dx = (xmax-xmin) / (N+1);
    x  = (xmin + dx:dx:xmax - dx)';
    hx = dx;
    hy = dx;
    y = x;
    nt = fix(Tf/dt);
    dt = Tf/nt;
    %
    % Set the initial condition(s).
    %
    U_ex = sin(pi * x);
    V_ex = sin(pi * y);
    S_ex = norm(U_ex, 2) * norm(V_ex, 2);
    U_ex = U_ex ./ norm(U_ex, 2);
    V_ex = V_ex ./ norm(V_ex, 2);
    e = ones(N, 1);
    A_direct = spdiags([e -2*e e], -1:1, N, N);
    f = (1 + 2i * pi^2) * U_ex * S_ex * V_ex';
    f_vec = f(:);
    L = kron(A_direct, speye(N, N)) + kron(speye(N, N), A_direct);
    BWD = speye(N * N, N * N) - 0.5*dt*(1i/dx^2) * L;
    FWD = speye(N * N, N * N) + 0.5*dt*(1i/dx^2) * L;

    U_f = U_ex;
    V_f = V_ex;
    S_f = (1 + 2i * pi^2) * S_ex;
    % Initialize time stepping right hand side
    U = U_f;
    V = V_f;
    S = S_f;

    u_vec = U*S*V';
    u_vec = u_vec(:);
    % u_vec_i holds the integral, intialize first point
    u_vec_i = 0.5*dt*u_vec;
    % Do some timestepping
    for it = 1:nt
        t = (it-1)*dt;
        u_vec = BWD\(FWD*u_vec);
        u_vec_i = u_vec_i + dt*exp(-(t+dt))*u_vec;
    end
    U_OFT = reshape(u_vec_i, N, N);
end