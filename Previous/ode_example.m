clear; clc; close all;

tau = 0.01;
q = 3;

n = 100;
h = 2 / (n + 1);
xx = -1 + h * (0:n+1)';
x = xx(2:n+1);

e = ones(n, 1);
A = (1 / h^2) * spdiags([e -2*e e], -1:1, n, n);

b = exp(-x.^2 / 0.1^2);

y = expm(1i * A * tau) * b;

n_t = 100;

[y_bar, z] = ode_prop(A, tau, b, n_t, q);

stability_inequality = 1;
for i = 1:length(z)
    stability_inequality = stability_inequality + z(i);
end

k = linspace(-4, 4, n);
l = linspace(-4, 4, n);
[X, Y] = meshgrid(k, l);
Z = X + 1i * Y;

% 2nd order Runge-Kutta method
RK2 = @(y) (1 + y + 1/2*y.^2);

% 3rd order Runge-Kutta method
RK3 = @(y) (1 + y + 1/2*y.^2 + 1/6*y.^3);

% 4th order Runge-Kutta method
RK4 = @(y) (1 + y + 1/2*y.^2 + 1/6*y.^3 + 1/24*y.^4);

region_2 = abs(RK2(Z)); region_3 = abs(RK3(Z)); region_4 = abs(RK4(Z));

figure(1)
subplot(2, 2, 3)
hold on
grid on
if q == 1
    % need circle
end
if q == 2
contour(X, Y, region_2, [1, 1], 'R');
end
if q == 3
contour(X, Y, region_3, [1, 1], 'R');
end
if q == 4
contour(X, Y, region_4, [1, 1], 'R');
end
plot(0, stability_inequality, '.', 'MarkerSize', 20, 'Color', [0, 0, 0]);
hold off;

subplot(2, 2, [1, 2])
plot(x, real(y), 'R' , x, imag(y), 'K', x, real(y_bar), 'R--', x, imag(y_bar), 'K--', 'LineWidth', 2);


function [y, z] = ode_prop(A, tau, y_0, n_t, q)
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
