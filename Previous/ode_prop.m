function y = ode_prop(A, tau, y_0, n_t, q)
    % Time step
    dt = tau / n_t;
    y = y_0;
    % For each time step
    for i_t = 1:n_t
        % initial df is dt
        df = dt;
        % new y = old y
        y_new = y;
        for i_tay = 1:q
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