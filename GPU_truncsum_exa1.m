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
  Tf   = 5.0;
  CFL  = 0.1;
  dt   = CFL * dx^2; 

  nmax = fix( Tf / dt );

  u   = gpuArray(zeros(N, N));
  um0 = gpuArray(zeros(N, N));
  um1 = gpuArray(zeros(N, N));
  vAp = gpuArray(zeros(N, N));
  vEx = gpuArray(zeros(N, N));
  X = gpuArray(zeros(N, N));
  Y = gpuArray(zeros(N, N));

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

  u   = (1 + 2i * pi^2) .* sin( pi * X ) .* sin(pi * Y);
  vAp = 0.5 * dt * u;

  um0 = u;  % aka u^{n-0}

  n = 1;

  r = 1i * dt / dx^2;

  e = ones(N, 1);
  A = gpuArray(spdiags([e -2*e e], -1:1, N, N));

  % u( 2:N-1 ) = u( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
  u = u + r * (A * u + u * A);

  vAp = vAp + dt * exp( -n * dt ) * u;

  um1 = um0;
  um0 = u;

  [U, S, V] = svd(u);
  [Um1, Sm1, Vm1] = svd(um1);
  [Um0, Sm0, Vm0] = svd(um0);

%
% For n ≥ 2, use a centered scheme (Leapfrog) for time and CS for space.
%
tic
  r = 2 * 1i * dt / dx^2;

  for n = 2:nmax
 
    % u = um1 + r * (A * u + u * A);
    % u( 2:N-1 ) = um1( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );

    U_hat = gpuArrau([Um1, A * U, U]);
    S_hat = gpuArray(blkdiag(Sm1, r * S, r * S));
    V_hat = gpuArray([Vm1, V, A * V]);
    cell = {U_hat, S_hat, V_hat};

    [U, S, V] = truncsum_fixed(cell, 1e-3, 100);
 
  %
  % Update OFT sum.
  %

  vAp = vAp + dt .* exp( -n * dt ) .* (U * S * V');

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
  vAp = gather(vAp);


toc

%
% Print relative error.
%
  vEx = sin( pi * X ) .* sin(pi * Y);  
  relErr = norm( vEx - vAp, 'inf' ) / norm( vEx, 'inf' );

  fprintf('\n' )
  fprintf(' Relative error = %8.2e\n', relErr )
  fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
