%  -----------------------------------------------------------------------------
%                                  LROFT2.m
%  -----------------------------------------------------------------------------
%  Solution of a BVP (an ODE) over [0,1] via the Operator Fourier Transform (OFT).
%
%  Author: Jack Kelley, Daniel Appelo, Edwin Jimenez, and Max Cubillos
%  Modified: 2/2/25
%  -----------------------------------------------------------------------------

addpath('service/service');

N    = 100; % # of spatial domain points.
xmin = 0.0; xmax = 1.0;
dx   = (xmax-xmin) / (N+1);
x    = (xmin + dx:dx:xmax - dx)';
y = x;
Tf   = 6;
dt = 0.01;
nt_max = fix(Tf / dt);
% Truncation at 10 x machine eps for the predicted space.
TOL_PRE = 10*2.2204e-16;
max_rank = 100;

use_truncation = 0;
use_direct = 1;

dlra_core_tol = 1e-12;
dlra_bug_tol = 1e-8;

hx = 2/(N+1);
hy = 2/(N+1);

C0 = 1/10;
C1 = 1;
C2 = 1;

u   = zeros( N, N );
um0 = zeros( N, N );
um1 = zeros( N, N );
vAp = zeros( N, N );
X = zeros(N, N);
Y = zeros(N, N);

% Store rank at each time step
ranks = zeros(1, nt_max);

for j = 1:N
    for i = 1:N
        X(i, j) = x(i);
        Y(i, j) = x(j);
    end
end

%
% Set up operator
%
e = ones(N, 1);
A1 = (1i/hx^2)*spdiags([e, -2 * e, e], -1:1,N,N);
B1 = speye(N, N);

e = ones(N, 1);
B2 = (1i/hy^2)*spdiags([e, -2 * e, e], -1:1,N,N);
A2 = speye(N, N);

RH_OP = {A1, B1
    A2, B2};

n_ops = size(RH_OP,1);

%
% Set the initial condition(s).
%

U = sin(pi * x);
V = sin(pi * y);
S = (1 + 2i * pi^2) * norm(U, 2) * norm(V, 2);
U = U ./ norm(U, 2);
V = V ./ norm(V, 2);
current_rank = 1;

% Do a direct solve
A_direct = spdiags([e -2*e e], -1:1, N, N);
f = U * S * V';
f = f(:);
L = kron(A_direct, speye(N, N)) + kron(speye(N, N), A_direct);
L = speye(N * N, N * N) - (1i / dx^2) * L;
U_direct = L \ f;
U_direct = reshape(U_direct, N, N);

% USV holds the integral, intialize first point
U_vAp = U;
V_vAp = V;
S_vAp = (dt / (2i * pi^2)) * S;

tic

ranks(1) = current_rank;

h = x(2)-x(1);
TOL_RES = C2*(dt^2+h^3)/h; %C2*dt^2;
TOL1 = C1*dt;
TOL2 = C2*(dt^2+h^3)/h;

% Do some timestepping
it = 0;
t = 0;
while t < Tf
    it = it+1;
    if (t+dt > Tf)
        dt = Tf-t;
    end

    % Prediction step.
    % Here we compute the column and row spaces
    % Additions to the subspaces based on
    % u_t = PDE
    if (use_truncation)
        for iops = 1:n_ops
            C{iops,1} = RH_OP{iops,1}*U;
            C{iops,2} = S;
            C{iops,3} = RH_OP{iops,2}*V;
        end
        [QU,QS,QV] = truncsum(C,TOL1,n_ops*current_rank);
        AU = [U QU];
        AV = [V QV];
    else
        AU = zeros(N,(n_ops+1)*current_rank);
        AV = zeros(N,(n_ops+1)*current_rank);
        AU(:,1:current_rank) = U;
        AV(:,1:current_rank) = V;
        for iops = 1:n_ops
            AU(:,iops*current_rank+1:(iops+1)*current_rank) = RH_OP{iops,1}*U;
            AV(:,iops*current_rank+1:(iops+1)*current_rank) = RH_OP{iops,2}*V;
        end
    end
    % Start by the "extra" solve
    % Orthogonalize the proposed spaces
    [QU,RU,PU] = qr(AU,'econ');
    [QV,RV,PV] = qr(AV,'econ');
    % Prediction by SVD columns space for U and V
    sd = abs(diag(RU));
    ru = length(find(sd >= TOL_PRE));
    sd = abs(diag(RV));
    rv = length(find(sd >= TOL_PRE));
    % These are in the weighted space
    Upre = QU(:,1:ru);
    Vpre = QV(:,1:rv);
    % Compute terms in the PDE based on the
    % predicted subspaces
    % This is the implicit Euler step.
    % Galerkin evolution
    Csylv = (Upre'*U)*(S)*(V'*Vpre);

    AT = (eye(ru,ru)-dt*(Upre'*(RH_OP{1,1}*Upre)));
    BT = -dt * ((RH_OP{2,2}*Vpre)'*Vpre);
    CORE = sylvester(AT,BT,Csylv);

    [Ucore,Score,Vcore] = svd(CORE);
    % We truncate the Galerkin evolution based on the LTE
    sd = diag(Score);
    energy = cumsum(sd(end:-1:1).^2);
    rnew = length(energy) - length(find(energy < TOL2^2));
    Unew = Upre*Ucore(:,1:rnew);
    Vnew = Vpre*Vcore(:,1:rnew);
    Snew = Score(1:rnew,1:rnew);

    % Check the norm of the residual
    CRES = cell(n_ops+2,3);
    % This is AXB^T
    for iops = 1:n_ops
        CRES{iops,1} = RH_OP{iops,1}*Unew;
        CRES{iops,2} = dt*Snew;
        CRES{iops,3} = RH_OP{iops,2}*Vnew;
    end
    CRES{n_ops+1,1} = Unew;
    CRES{n_ops+1,2} = -Snew;
    CRES{n_ops+1,3} = Vnew;
    CRES{n_ops+2,1} = U;
    CRES{n_ops+2,2} = S;
    CRES{n_ops+2,3} = V;
    % We truncate slightly tighter than for the solver
    [rU,rS,rV] = truncsum(CRES,0.1*TOL2,n_ops*rnew);
    res = sqrt(inner_low(rU,rS,rV,rU,rS,rV));

    % If the residual is small enough we break out
    % of the while loop with a new timestep
    % If not, we use don't adjust the timestep yet
    % but first try to add the BUG space to see if
    % that reduces the residual suficiently.
    if res < TOL_RES
        U = Unew;
        V = Vnew;
        S = Snew;
        current_rank = rnew;
    else

        % Then we add the BUG spaces
        K0 = U*S;
        nk = size(K0,2);
        CK = cell(n_ops+1,2);
        for iops = 1:n_ops
            CK{iops,1} = RH_OP{iops,1};
            CK{iops,2} = -dt*(RH_OP{iops,2}*V)'*V;
        end
        CK{n_ops+1,1} = speye(N,N);
        CK{n_ops+1,2} = speye(nk,nk);
        [K1,FLAG,RELRES,ITER] = gmres_sylvester(K0,CK, dlra_bug_tol,N*nk);

        L0 = V*S';
        nl = size(L0,2);
        CL = cell(n_ops+1,2);
        for iops = 1:n_ops
            CL{iops,1} = RH_OP{iops,2};
            CL{iops,2} = -dt*(RH_OP{iops,1}*U)'*U;
        end
        CL{n_ops+1,1} = speye(N,N);
        CL{n_ops+1,2} = speye(nk,nk);
        [L1,FLAG,RELRES,ITER] = gmres_sylvester(L0,CL, dlra_bug_tol, N*nl);
        % Merge the spaces
        AU = [AU K1];
        AV = [AV L1];

        % Orthogonalize the proposed spaces
        [QU,RU,PU] = qr(AU,'econ');
        [QV,RV,PV] = qr(AV,'econ');
        % Prediction by SVD columns space for U and V
        sd = abs(diag(RU));
        ru = length(find(sd >= TOL_PRE));
        sd = abs(diag(RV));
        rv = length(find(sd >= TOL_PRE));
        % These are in the weighted space
        Upre = QU(:,1:ru);
        Vpre = QV(:,1:rv);
        % Compute terms in the PDE based on the
        % predicted subspaces
        % This is the implicit Euler step.
        % Galerkin evolution

        Csylv = (Upre'*U)*S*(V'*Vpre);

        AT = (eye(ru,ru)-dt*(Upre'*(RH_OP{1,1}*Upre)));
        BT = -dt * ((RH_OP{2,2}*Vpre)'*Vpre);
        CORE = sylvester(AT,BT,Csylv);

        [Ucore,Score,Vcore] = svd(CORE);

        % We truncate the Galerkin evolution based on the LTE
        sd = diag(Score);
        energy = cumsum(sd(end:-1:1).^2);
        rnew = length(energy) - length(find(energy < TOL2^2));
        U = Upre*Ucore(:,1:rnew);
        V = Vpre*Vcore(:,1:rnew);
        S = Score(1:rnew,1:rnew);
        current_rank = rnew;
    end

    %
    % Update OFT integral -------------------------------------------------
    %
    C = {U_vAp, S_vAp, V_vAp
        U, (dt / (2i * pi^2)) * exp(-it * dt) * S, V};

    [U_vAp, S_vAp, V_vAp] = truncsum(C, dlra_core_tol, max_rank);
    ranks(it) = current_rank;

    t = t+dt;
end
toc

%
% Print relative error.
%
vEx = sin(pi * X) .* sin(pi * Y);
relErr = norm( vEx - (abs(U_vAp * S_vAp * V_vAp')), 'fro' ) / norm( vEx, 'fro');
relErr2 = norm( U_direct - (abs(U_vAp * S_vAp * V_vAp')), 'fro' ) / norm( U_direct, 'fro');

fprintf('\n' )
fprintf(' Relative error = %8.2e\n', relErr)
fprintf(' Direct error = %8.2e\n', relErr2)
fprintf('\n' )


%  -----------------------------------------------------------------------------
%  -----------------------------------------------------------------------------
