function [U_vAp, S_vAp, V_vAp,ranks]=LROFT_LRIAT2(N,dt,Tf);
    
    addpath('/service');
    xmin = -1.0;
    xmax =  1.0;
    dx = (xmax-xmin) / (N+1);
    x  = (xmin + dx:dx:xmax - dx)';
    hx = dx;
    hy = dx;
    y = x;
    nt = fix(Tf/dt);
    dt = Tf/nt;

    % Truncation at 10 x machine eps for the predicted space.
    TOL_PRE = 10*2.2204e-16;
    max_rank = 100;
    dlra_core_tol = 1e-12;
    dlra_bug_tol = 1e-8;

    C0 = 1e-1;
    C1 = 1e-0;
    C2 = 1e-0;

    vAp = zeros( N, N );
    X = zeros(N, N);
    Y = zeros(N, N);

    % Store rank at each time step
    ranks = zeros(1, nt);

    %
    % Set up operator
    %
    e = ones(N, 1);
    A1 = (1i/hx^2)*spdiags([e, -2 * e, e], -1:1,N,N);
    B1 = speye(N, N);

    A2 = speye(N, N);
    B2 = (1i/hy^2)*spdiags([e, -2 * e, e], -1:1,N,N);
    
    A1 = full(A1);
    B2 = full(B2);
    RH_OP = {A1, B1'
             A2, B2'};
    n_ops = size(RH_OP,1);
    %
    % Set the initial condition(s).
    %
    current_rank = 1;
    U_ex = sin(pi * x);
    V_ex = sin(pi * y);
    S_ex = norm(U_ex, 2) * norm(V_ex, 2);
    U_ex = U_ex ./ norm(U_ex, 2);
    V_ex = V_ex ./ norm(V_ex, 2);

    U_f = U_ex;
    V_f = V_ex;
    S_f = (1 + 2i * pi^2) * S_ex;
    % Initialize time stepping right hand side
    U = U_f;
    V = V_f;
    S = S_f;
    % USV holds the integral, intialize first point
    U_vAp = U;
    V_vAp = V;
    S_vAp = 0.5*dt * S;

    ranks(1) = current_rank;

    h = x(2)-x(1);
    TOL_RES = C2*(dt^3+h^3)/h; 
    TOL1 = C1*dt^2;
    TOL2 = C2*(dt^3+h^3)/h;
    
    % Do some timestepping
    for it = 1:nt
        t = (it-1)*dt;
        % Prediction step.
        % Here we compute the column and row spaces
        % Additions to the subspaces based on
        % u_t = PDE
        AU = zeros(N,(n_ops+1)*current_rank);
        AV = zeros(N,(n_ops+1)*current_rank);
        AU(:,1:current_rank) = U;
        AV(:,1:current_rank) = V;
        for iops = 1:n_ops
            AU(:,iops*current_rank+1:(iops+1)*current_rank) = RH_OP{iops,1}*U;
            AV(:,iops*current_rank+1:(iops+1)*current_rank) = RH_OP{iops,2}*V;
        end
        AU = [AU RH_OP{1,1}*RH_OP{1,1}*U];
        AV = [AV RH_OP{2,2}*RH_OP{2,2}*V];
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
        % This is the Trapezoidal step.
        % Galerkin evolution
        Csylv = Upre'*(U*S*V' + 0.5*dt*RH_OP{1,1}*U*S*V' + 0.5*dt*U*S*(RH_OP{2,2}*V)')*Vpre;
        AT = (0.5*eye(ru,ru)-0.5*dt*(Upre'*(RH_OP{1,1}*Upre)));
        BT = (0.5*eye(rv,rv)-0.5*dt*((RH_OP{2,2}*Vpre)'*Vpre));
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
        CRES = cell(6,3);
        % This is AXB^T
        CRES{1,1} = RH_OP{1,1}*U;
        CRES{1,2} = -0.5*dt*S;
        CRES{1,3} = RH_OP{1,2}*V;
        CRES{2,1} = RH_OP{2,1}*U;
        CRES{2,2} = -0.5*dt*S;
        CRES{2,3} = RH_OP{2,2}*V;
        CRES{3,1} = RH_OP{1,1}*Unew;
        CRES{3,2} = -0.5*dt*Snew;
        CRES{3,3} = RH_OP{1,2}*Vnew;
        CRES{4,1} = RH_OP{2,1}*Unew;
        CRES{4,2} = -0.5*dt*Snew;
        CRES{4,3} = RH_OP{2,2}*Vnew;
        CRES{5,1} = Unew;
        CRES{5,2} = Snew;
        CRES{5,3} = Vnew;
        CRES{6,1} = U;
        CRES{6,2} = -S;
        CRES{6,3} = V;
        % We truncate slightly tighter than for the solver
        [rU,rS,rV] = truncsum(CRES,0.1*TOL2,N);
        res = sqrt(abs(inner_low(rU,rS,rV,rU,rS,rV)));
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
            K1 = sylvester((-dt*RH_OP{1, 1}),eye(nk, nk)-dt*(RH_OP{2, 2}*V)'*V,K0);
            L0 = V*S';
            nl = size(L0,2);
            L1 = sylvester(eye(nk, nk)-dt*(RH_OP{1,1}*U)'*U, (-dt*RH_OP{2, 2}),L0');
            % Merge the spaces
            AU = [AU K1];
            AV = [AV L1'];
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
            % This is the Trapezoidal step.
            % Galerkin evolution
            Csylv = Upre'*(U*S*V' + 0.5*dt*RH_OP{1,1}*U*S*V' + 0.5*dt*U*S*(RH_OP{2,2}*V)')*Vpre;
            AT = (0.5*eye(ru,ru)-0.5*dt*(Upre'*(RH_OP{1,1}*Upre)));
            BT = (0.5*eye(rv,rv)-0.5*dt*((RH_OP{2,2}*Vpre)'*Vpre));
            CORE = sylvester(AT,BT,Csylv);
            [Ucore,Score,Vcore] = svd(CORE);

            % This is the implicit Euler step.
            % Galerkin evolution
            %Csylv = (Upre'*U)*S*(V'*Vpre);
            %AT = 0.5*eye(ru,ru)-dt*(Upre'*(RH_OP{1,1}*Upre));
            %BT = 0.5*eye(rv,rv)-dt*((RH_OP{2,2}*Vpre)'*Vpre);
            %CORE = sylvester(AT,BT,Csylv);
            %[Ucore,Score,Vcore] = svd(CORE);

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
        % Icrement OFT integral -------------------------------------------------
        %
        C = {U_vAp,S_vAp,V_vAp
             U,dt*exp(-(t+dt))*S,V};
        [U_vAp, S_vAp, V_vAp] = truncsum(C, dlra_core_tol, max_rank);
        ranks(it) = current_rank;
    end
end