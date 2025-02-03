function [U,S,V,RRR,TTT] = bug_ie(tend,nx,ny,nsteps,scrpt);

    addpath('./../../service');
    eval(scrpt);
    
    % Truncation at 10 x machine eps for the predicted space.
    TOL_PRE = 10*2.2204e-16;
    
    % Store the singular values, the rank, error and time.
    RRR = zeros(nsteps+1,4);
    TTT = zeros(nsteps+1,1);

    t = 0;
    % Initial data
    % If a low rank representation is known we use it 
    % If only a 2D function is known we compute the SVD (could be
    % improved to use threshold SVD)
    if (r > 0) 
        U = zeros(nx,r);
        V = zeros(ny,r);
        for ir = 1:r
            for i = 1:nx
                U(i) = gx(x(i),ir);
            end
            for j = 1:ny
                V(j) = gy(y(j),ir);
            end
        end
        [QU,RU] = qr(U,'econ');
        [QV,RV] = qr(V,'econ');
        [U1,S1,V1] = svd(RU*RV');
        U = QU*U1(:,1:r);
        V = QV*V1(:,1:r);
        S = S1(1:r,1:r);
    else
        T = zeros(nx,ny);
        % Initial data and forcing
        for i = 1:nx
            for j = 1:ny
                T(i,j) = g(x(i),y(j));
            end
        end
        [U,S,V] = svd(T);
        % Truncate based on the Frobenius norm
        % We sum up the squares of the singular values from below
        % and compare to TOL^2
        sd = diag(S);
        energy = cumsum(sd(end:-1:1).^2);
        r = length(energy) - length(find(energy < dt_max^6));
        U = U(:,1:r);
        V = V(:,1:r);
        S = S(1:r,1:r);
    end
    RRR(1,1) = r;

    n_ops = size(RH_OP,1);
    % The tolerances for truncation and rank rounding.
    dt = tend/nsteps;
    TOL1 = C1*dt;
    TOL2 = C2*dt^2;

    % Do some timestepping
    t = 0;
    for it = 1:nsteps
        t = (it-1)*dt;

        % BUG method
        K0 = U*S;
        nk = size(K0,2);
        CK = cell(n_ops+1,2);
        for iops = 1:n_ops
            CK{iops,1} = RH_OP{iops,1};
            CK{iops,2} = -dt*(RH_OP{iops,2}*V)'*V;
        end
        CK{n_ops+1,1} = speye(nx,nx);
        CK{n_ops+1,2} = speye(nk,nk);
        [K1,FLAG,RELRES,ITER] = gmres_sylvester(K0,CK, dlra_bug_tol,nx*nk);
        RRR(it+1,2) = ITER(2);
        L0 = V*S';
        nl = size(L0,2);
        CL = cell(n_ops+1,2);
        for iops = 1:n_ops
            CL{iops,1} = RH_OP{iops,2};
            CL{iops,2} = -dt*(RH_OP{iops,1}*U)'*U;
        end
        CL{n_ops+1,1} = speye(ny,ny);
        CL{n_ops+1,2} = speye(nk,nk);
        [L1,FLAG,RELRES,ITER] = gmres_sylvester(L0,CL, dlra_bug_tol, ny*nl);
        RRR(it+1,3) = ITER(2);
        % Merge the spaces
        AU = [U K1];
        AV = [V L1];
        
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
        CGE = cell(n_ops+1,2);
        for iops = 1:n_ops
            CGE{iops,1} = -dt*(Upre'*(RH_OP{iops,1}*Upre));
            CGE{iops,2} = (RH_OP{iops,2}*Vpre)'*Vpre;
        end
        CGE{n_ops+1,1} = speye(ru,ru);
        CGE{n_ops+1,2} = speye(rv,rv);
        [CORE,FLAG,RELRES,ITER] = gmres_sylvester(Csylv,CGE, dlra_core_tol, max(size(Csylv)));
        RRR(it+1,4) = ITER(2);
        [Ucore,Score,Vcore] = svd(CORE);
        % We truncate the Galerkin evolution based on the LTE
        sd = diag(Score);
        energy = cumsum(sd(end:-1:1).^2);
        rnew = length(energy) - length(find(energy < TOL2^2));
        Unew = Upre*Ucore(:,1:rnew);
        Vnew = Vpre*Vcore(:,1:rnew);
        Snew = Score(1:rnew,1:rnew);
        % Record the rank and the time
        RRR(it+1,1) = r;
        TTT(it+1) = t+dt;
    end
end
