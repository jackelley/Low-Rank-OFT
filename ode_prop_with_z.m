function [y, z] = ode_prop_with_z(A, tau, y_0, n_t, q)
    % Time step
    dt = tau / n_t;
    y = y_0;
    z = zeros(q, 1);
    D = eig(A);
    % For each time step
    for i_t = 1:n_t
        % initial df is dt
        df = dt;
        % new y = old y
        y_new = y;
        for i_tay = 1:q
            z(i_tay) = D(i_tay) * (dt^i_tay / factorial(i_tay));
            % y prime is iAy
            y_p = 1i * A * y;
            % Add y prime to Taylor expansion
            y_new = y_new + (df * y_p);
            % compute new scalar multiple
            df = df * dt / (i_tay + 1);
            % y = y prime so we can keep differentiating
            y = y_p;
        end
        y = y_new;
    end
end