dlra_core_tol = 1e-12;
dlra_bug_tol = 1e-8;
use_truncation = 0;
use_direct = 1;

C0 = 1/10;
C1 = 1;
C2 = 1;

r = 1;
gx = @(x,ir) sin(1*pi*x);
gy = @(y,ir) sin(1*pi*y);
g = @(x,y) gx(x,1).*gy(y,1);

hx = 2/(nx+1);
hy = 2/(ny+1);

x = -1 + (1:nx)'*hx;
y = -1 + (1:ny)'*hy;
xb = [-1;x;1];
yb = [-1;y;1];

% Set up mesh
X = zeros(nx,ny);
Y = zeros(nx,ny);
for i = 1:nx
    for j = 1:ny
        X(i,j) = x(i);
        Y(i,j) = y(j);
    end
end
n = nx*ny;
X_vector = reshape(X,n,1);
Y_vector = reshape(Y,n,1);

a1 = @(x) 1+0.*sin(pi*x);
a2 = @(x) 0.3+0.0*sin(pi*x);
a3 = @(x) 0.3+0.0*cos(pi*x);
a4 = @(x) 1+0.*sin(pi*x);

b1 = @(y) 1+0.*cos(pi*y);
b2 = @(y) 0.3+0.0*cos(pi*y);
b3 = @(y) 0.3+0.0*sin(pi*y);
b4 = @(y) 1+0.*cos(pi*y);

% Point-wise matrices for all but a1 and b4
% Averages for those so we construct the 
% difference opperator right away 

a1b = a1(xb);
a1a = a1(x);
a1m = 0.5*(a1b(3:nx+2)+a1b(2:nx+1));
a10 = -0.5*(a1b(3:nx+2)+2*a1b(2:nx+1)+a1b(1:nx));
a1p = [0; 0.5*(a1b(3:nx+1)+a1b(2:nx))];
D2A1 = (1/hx/hx)*spdiags([a1m a10 a1p],-1:1,nx,nx);

A2 = spdiags(a2(x),0,nx,nx);
A3 = spdiags(a3(x),0,nx,nx);
A4 = spdiags(a4(x),0,nx,nx);

B1 = spdiags(b1(y),0,ny,ny);
B2 = spdiags(b2(y),0,ny,ny);
B3 = spdiags(b3(y),0,ny,ny);

b4b = b4(yb);
b4m = 0.5*(b4b(3:ny+2)+b4b(2:ny+1));
b40 = -0.5*(b4b(3:ny+2)+2*b4b(2:ny+1)+b4b(1:ny));
b4p = [0; 0.5*(b4b(3:ny+1)+b4b(2:ny))];
D2B4 = (1/hy/hy)*spdiags([b4m b40 b4p],-1:1,ny,ny);

% Centered second order operators
e = ones(nx,1);
D0X = (0.5/hx)*spdiags([-e 0*e e], -1:1, nx, nx);
e = ones(ny,1);
D0Y = (0.5/hy)*spdiags([-e 0*e e], -1:1, ny, ny);

Inx = speye(nx,nx);
Iny = speye(ny,ny);


ROTX = -spdiags(x,0,nx,nx);
ROTY = spdiags(y,0,ny,ny);

AA1 = D2A1;
BB1 = B1;
AA2 = D0X*A2;
BB2 = B2*D0Y;
AA3 = A3*D0X;
BB3 = D0Y*B3;
AA4 = A4;
BB4 = D2B4;

RH_OP = {AA1,BB1
         AA2,BB2
         AA3,BB3
         AA4,BB4};