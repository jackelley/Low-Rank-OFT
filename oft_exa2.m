%  -----------------------------------------------------------------------------
%                                   oft_exa1.m
%  -----------------------------------------------------------------------------
%  Solution of a BVP (an ODE) over (-oo,+oo) via the Operator Fourier Transform (OFT).
%
%  Author: Edwin Jimenez and Max Cubillos
%  Modified: 17 September 2024
%  -----------------------------------------------------------------------------

  N    = 1000; % # of spatial domain points.
  xmin = -5.0; xmax = 5.0;
  dx   = (xmax-xmin) / (N-1);
  x    = xmin:dx:xmax;
  Tf   = 5.0;
  CFL  = 0.1;
  dt   = CFL * dx; 

  nmax = fix( Tf / dt );

  u   = zeros( N, 1 );
  w   = zeros( N, 1 );
  vAp = zeros( N, 1 );
  vEx = zeros( N, 1 );

%
% Set the initial condition(s). 
%
  n = 0;

  u   = (3 - 4 * x.^2) .* exp( -x.^2 );
  w   = (3 - 4 * x.^2) .* exp( -x.^2 );
  vAp = 0.25 * dt * ( u + w );

%
% For n ≥ 1, use upwinding scheme.
%
  r = dt / dx;

  for n = 1:nmax
 
    u( 1:N-1 ) = u( 1:N-1 ) + r * ( u( 2:N ) - u( 1:N-1 ) );
    w( 2:N )   = w( 2:N )   - r * ( w( 2:N ) - w( 1:N-1 ) );

  %
  % Update OFT sum.
  %
    vAp = vAp + 0.5 * dt * exp( -n * dt ) .* ( u + w );

  end

%
% Print relative error.
%
  vEx = exp( -x.^2 );  

  relErr = norm( vEx - vAp, 'inf' ) / norm( vEx, 'inf' );

  fprintf('\n' )
  fprintf(' Relative error = %8.2e\n', relErr )
  fprintf('\n' )

 
%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
