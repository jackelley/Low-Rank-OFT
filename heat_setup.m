dlra_core_tol = 1e-12;
dlra_bug_tol = 1e-8;
use_truncation = 0;
use_direct = 1;

C0 = 1/10;
C1 = 1;
C2 = 1;

r = 1;
gx = @(x,ir) exp(-36 * x^2);
gy = @(y,ir) exp(-36 * y^2);
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

e = ones(nx, 1);
A1 = (1i/hx/hx)*spdiags([e, -2 * e, e], -1:1,nx,nx);
B1 = speye(ny, ny);

e = ones(ny, 1);
B2 = (1i/hy/hy)*spdiags([e, -2 * e, e], -1:1,ny,ny);
A2 = speye(nx, nx);

RH_OP = {A1, B1
         A2, B2};