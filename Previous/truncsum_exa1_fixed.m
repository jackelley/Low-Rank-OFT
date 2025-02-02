%  -----------------------------------------------------------------------------
%                                   oft_exa1.m
%  -----------------------------------------------------------------------------
%  Solution of a BVP (an ODE) over [0,1] via the Operator Fourier Transform (OFT).
%
%  Author: Edwin Jimenez and Max Cubillos
%  Modified: 17 September 2024
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

  nmax = fix( Tf / dt );

  u   = zeros( N, N );
  um0 = zeros( N, N );
  um1 = zeros( N, N );
  vAp = zeros( N, N );
  vEx = zeros( N, N );
  X = zeros(N, N);
  Y = zeros(N, N);

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
  n = 0;

  % u = (1 + 2i * pi^2) .* sin( pi * X ) .* sin(pi * Y);
  U = (1 + 2i * pi^2) .* sin(pi * x);
  V = sin(pi * y);
  S = norm(U, 2) * norm(V, 2);
  U = U ./ norm(U, 2);
  V = V ./ norm(V, 2);

  % Do a direct solve
  f = U * S * V';
  f = f(:);
  L = kron(A, speye(N, N)) + kron(speye(N, N), A);
  L = speye(N * N, N * N) - (1i / dx^2) * L;
  U_direct = L \ f;
  U_direct = reshape(U_direct, N, N);
  % mesh(X, Y, abs(U_direct));
  
  % USV holds the integral, intialize first point
  U_vAp = U;
  V_vAp = V;
  S_vAp = 0.5 * dt * S;

  % vAp = 0.5 * dt * u;

  % um0 = u;  % aka u^{n-0}
  Um0 = U;
  Vm0 = V;
  Sm0 = S;
  
  n = 1;

  r = 1i * dt / dx^2;

  % u( 2:N-1 ) = u( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
  % u = u + r * (A * u + u * A);

  U_hat = [U, A * U, U];
  S_hat = blkdiag(S, r * S, r * S);
  V_hat = [V, V, A * V];
  cell = {U_hat, S_hat, V_hat};

  [U, S, V] = truncsum_fixed(cell, tol, max_rank);

  C = {U_vAp, S_vAp, V_vAp
      U, dt * exp( -n * dt ) * S, V};

  % Update integral
  [U_vAp, S_vAp, V_vAp] = truncsum_fixed(C, tol, max_rank);

  % vAp = vAp + dt * exp( -n * dt ) * u;

  % um1 = um0;
  % um0 = u;

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

  for n = 2:nmax
 
    % u = um1 + r * (A * u + u * A);
    % u( 2:N-1 ) = um1( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
    
    % Take one time step
    C = {Um1, Sm1, Vm1
        A * U, r * S, V
        U, r * S, A * V};

    [U, S, V] = truncsum_fixed(C, tol, max_rank);

    

  % Update OFT sum
  C = {U_vAp, S_vAp, V_vAp
      U, dt * exp( -n * dt ) * S, V};

  [U_vAp, S_vAp, V_vAp] = truncsum_fixed(C, tol, max_rank);

  %
  % Update previous solutions.
  %
    %um1 = um0;
    %um0 = u;
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
  relErr = norm( vEx - (U_vAp * S_vAp * V_vAp'), 'fro' ) / norm( vEx, 'fro');
  relErr2 = norm( U_direct - (U_vAp * S_vAp * V_vAp'), 'fro' ) / norm( U_direct, 'fro');

  fprintf('\n' )
  fprintf(' Relative error = %8.2e\n', relErr)
  fprintf(' Direct error = %8.2e\n', relErr2)
  fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
