%  -----------------------------------------------------------------------------
%                                  LROFT1.m
%  -----------------------------------------------------------------------------
%  Solution of a BVP (an ODE) over [0,1] via the Operator Fourier Transform (OFT).
%
%  Author: Jack Kelley, Daniel Appelo, Edwin Jimenez, and Max Cubillos
%  Modified: 2/2/25
%  -----------------------------------------------------------------------------

N    = 100; % # of spatial domain points.
xmin = 0.0; xmax = 1.0;
dx   = (xmax-xmin) / (N+1);
x    = (xmin + dx:dx:xmax - dx)';
y = x;
Tf   = 6;
CFL  = 0.1;
dt   = CFL * dx^2;
tol = 1e-3;
max_rank = 100;

nt_max = fix( Tf / dt );

u   = zeros( N, N );
um0 = zeros( N, N );
um1 = zeros( N, N );
vAp = zeros( N, N );
X = zeros(N, N);
Y = zeros(N, N);

% Store rank at each time step
ranks = zeros(1, nt_max);

e = ones(N, 1);
A = spdiags([e -2*e e], -1:1, N, N);

for j = 1:N
    for i = 1:N
        X(i, j) = x(i);
        Y(i, j) = x(j);
    end
end

%
% Set the initial condition(s). For the 1st step, use FT CS scheme.
%

U = sin(pi * x);
V = sin(pi * y);
S = (1 + 2i * pi^2) * norm(U, 2) * norm(V, 2);
U = U ./ norm(U, 2);
V = V ./ norm(V, 2);

% Do a direct solve
f = U * S * V';
f = f(:);
L = kron(A, speye(N, N)) + kron(speye(N, N), A);
L = speye(N * N, N * N) - (1i / dx^2) * L;
U_direct = L \ f;
U_direct = reshape(U_direct, N, N);
return
  
% USV holds the integral, intialize first point
U_vAp = U;
V_vAp = V;
S_vAp = 0.5 * dt * S;

% Swap time level
Um0 = U;
Vm0 = V;
Sm0 = S;
  
nt = 1;

r = 1i * dt / dx^2;

U_hat = [U, A * U, U];
S_hat = blkdiag(S, r * S, r * S);
V_hat = [V, V, A * V];
cell = {U_hat, S_hat, V_hat};

[U, S, V] = truncsum(cell, tol, max_rank);

C = {U_vAp, S_vAp, V_vAp
    U, dt * exp( -nt * dt ) * S, V};

% Update integral
[U_vAp, S_vAp, V_vAp] = truncsum(C, tol, max_rank);

% Store rank
ranks(nt) = size(S_vAp, 1);

% Swap time levels
Um1 = Um0;
Vm1 = Vm0;
Sm1 = Sm0;

Um0 = U;
Vm0 = V;
Sm0 = S;

%
% For n ≥ 2, use a centered scheme (Leapfrog) for time and CS for space.
%
tic
r = 2 * 1i * dt / dx^2;

for nt = 2:nt_max
    
    % Take one time step
    C = {Um1, Sm1, Vm1
         A * U, r * S, V
         U, r * S, A * V};

    [U, S, V] = truncsum(C, tol, max_rank);

    % Update OFT sum
    C = {U_vAp, S_vAp, V_vAp
         U, dt * exp( -nt * dt ) * S, V};

    [U_vAp, S_vAp, V_vAp] = truncsum(C, tol, max_rank);

    % Store rank
    ranks(nt) = size(S_vAp, 1);
    
    %
    % Update previous solutions.
    %
    Um1 = Um0;
    Sm1 = Sm0;
    Vm1 = Vm0;

    Um0 = U;
    Sm0 = S;
    Vm0 = V;

end
toc

%
% Print relative error.
%
vEx = sin( pi * X ) .* sin(pi * Y); 
relErr = norm( vEx - (abs(U_vAp * S_vAp * V_vAp')), 'fro' ) / norm( vEx, 'fro');
relErr2 = norm( U_direct - (abs(U_vAp * S_vAp * V_vAp')), 'fro' ) / norm( U_direct, 'fro');

fprintf('\n' )
fprintf(' Relative error = %8.2e\n', relErr)
fprintf(' Direct error = %8.2e\n', relErr2)
fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
