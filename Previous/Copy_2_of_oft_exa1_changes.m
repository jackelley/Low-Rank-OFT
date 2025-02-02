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

  u   = zeros( N, N );
  um0 = zeros( N, N );
  um1 = zeros( N, N );
  vAp = zeros( N, N );
  vEx = zeros( N, N );
  X = zeros(N, N);
  Y = zeros(N, N);

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
  um1 = 0;  % aka u^{n-1}

  n = 1;

  r = 1i * dt / dx^2;

  e = ones(N, 1);
  A = spdiags([e -2*e e], -1:1, N, N);

  % u( 2:N-1 ) = u( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
  u = u + r * (A * u + u * A);

  vAp = vAp + dt * exp( -n * dt ) * u;

  um1 = um0;
  um0 = u;

%
% For n ≥ 2, use a centered scheme (Leapfrog) for time and CS for space.
%
tic
  r = 2 * 1i * dt / dx^2;

  for n = 2:nmax
 
    u = um1 + r * (A * u + u * A);
    % u( 2:N-1 ) = um1( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
 
  %
  % Update OFT sum.
  %
    vAp = vAp + dt .* exp( -n * dt ) .* u;

  %
  % Update previous solutions.
  %
    um1 = um0;
    um0 = u;

  end
  toc

%
% Print relative error.
%
  vEx = sin( pi * X ) .* sin(pi * Y);  
  relErr = norm( vEx - vAp, 'fro' ) / norm( vEx, 'fro' );

  fprintf('\n' )
  fprintf(' Relative error = %8.2e\n', relErr )
  fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
