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
  dx   = (xmax-xmin) / (N-1);
  x    = xmin:dx:xmax;
  Tf   = 5.0;
  CFL  = 0.1;
  dt   = CFL * dx^2; 

  nmax = fix( Tf / dt );

  u   = zeros( N, 1 );
  um0 = zeros( N, 1 );
  um1 = zeros( N, 1 );
  vAp = zeros( N, 1 );
  vEx = zeros( N, 1 );

%
% Set the initial condition(s). For the 1st step, use FT CS scheme.
%
  n = 0;

  u   = (1 + 1i * pi^2) * sin( pi * x );
  vAp = 0.5 * dt * u;

  um0 = u;  % aka u^{n-0}
  um1 = 0;  % aka u^{n-1}

  n = 1;

  r = 1i * dt / dx^2;

  u( 2:N-1 ) = u( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
  u( 1 ) = 0;
  u( N ) = 0;

  vAp = vAp + dt * exp( -n * dt ) * u;

  um1 = um0;
  um0 = u;

%
% For n ≥ 2, use a centered scheme (Leapfrog) for time and CS for space.
%
  r = 2 * 1i * dt / dx^2;

  for n = 2:nmax
 
    u( 2:N-1 ) = um1( 2:N-1 ) + r * ( u( 3:N ) - 2 * u( 2:N-1 ) + u( 1:N-2 ) );
    u( 1 )     = 0;
    u( N )     = 0;
 
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

%
% Print relative error.
%
  vEx = sin( pi * x );  
  relErr = norm( vEx - vAp, 'inf' ) / norm( vEx, 'inf' );

  fprintf('\n' )
  fprintf(' Relative error = %8.2e\n', relErr )
  fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
