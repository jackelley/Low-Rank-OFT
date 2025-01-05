clear; clc; close all;
% Time
start_time = 0;
dt_initial = 0.005;
dt_end = 10 * dt_initial;

L = 2;
kappa = 10;
a0 = 10;
T = kappa * L;

% Get times, n_t, and constants
[times, n_t, a, b] = compute_time_step_sizes(dt_initial, T, dt_end);

% Find time step sizes from times
time_steps = zeros(1, n_t);
time_steps(1) = dt_initial;
for i = 1:n_t - 1
    time_steps(i + 1) = times(i + 1) - times(i);
end

% Spatial
n_x = 200;
x = linspace(-1, 1, n_x);

g = exp(-a0 * abs(x).^2 + 1i * kappa * x(1));

W1 = w1(times(1:end-1), times(2:end));
W2 = w2(times(1:end-1), times(2:end));

weights = zeros(n_t, 1);
weights(1) = W1(1);
weights(2:n_t - 1) = W2(1:n_t - 2) + W1(2:n_t-1);
weights(n_t) = W2(n_t - 1);

k = [0:n_x/2-1, -n_x/2:-1]' * (2*pi/L);
g_hat = fft(g);
u = ifft(exp(((-1i .* k.^2 .* times) / kappa^2)) .* g_hat');
I = zeros(n_x, 1);
for i = 1:n_t
    I = I + weights(i) .* u(:, i);
end

I = I / sqrt(-1i / pi);

plot(x, real(I))

e = ones(n_x, 1);
A = (1/(kappa^2)) * spdiags([e -2*e e], -1:1, n_x, n_x);
C = (speye(n_x, n_x) + A)^(1/2);
v_exact = (speye(n_x, n_x) + A)^(1/2) \ g';

relative_error = norm(v_exact - I, 'inf') / norm(v_exact', 'inf');
