clear; clc; close all;

num_taus = 1000;
tau = linspace(0.000001, 0.001, num_taus);

q = 3;

n = 100;
h = 2 / (n + 1);
xx = -1 + h * (0:n+1)';
x = xx(2:n+1);
n_t = 100;
e = ones(n, 1);

A1 = -1;
A2 = 1i;
A3 = (1 / h^2) * spdiags([e -2*e e], -1:1, n, n);
A4 = (1i / h^2) * spdiags([e -2*e e], -1:1, n, n);

b = exp(-x.^2 / 0.1^2);

y1 = zeros(n, num_taus);
y2 = zeros(n, num_taus);
y3 = zeros(n, num_taus);
y4 = zeros(n, num_taus);
y_bar1 = zeros(n, num_taus);
y_bar2 = zeros(n, num_taus);
y_bar3 = zeros(n, num_taus);
y_bar4 = zeros(n, num_taus);

error1 = zeros(num_taus, 1);
error2 = zeros(num_taus, 1);
error3 = zeros(num_taus, 1);
error4 = zeros(num_taus, 1);


for i = 1:num_taus
    y1(:, i) = expm(1i * A1 * tau(i)) * b;
    y2(:, i) = expm(1i * A2 * tau(i)) * b;
    y3(:, i) = expm(1i * A3 * tau(i)) * b;
    y4(:, i) = expm(1i * A4 * tau(i)) * b;
    y_bar1(:, i) = ode_prop(A1, tau(i), b, n_t, q);
    y_bar2(:, i) = ode_prop(A2, tau(i), b, n_t, q);
    y_bar3(:, i) = ode_prop(A3, tau(i), b, n_t, q);
    y_bar4(:, i) = ode_prop(A4, tau(i), b, n_t, q);

    error1(i) = max(abs(real(y1(:, i)) - real(y_bar1(:, i))));
    error2(i) = max(abs(real(y2(:, i)) - real(y_bar2(:, i))));
    error3(i) = max(abs(real(y3(:, i)) - real(y_bar3(:, i))));
    error4(i) = max(abs(real(y4(:, i)) - real(y_bar4(:, i))));
end

figure(1)
subplot(2, 2, 1, 'XScale', 'log', 'YScale', 'log')
plot(tau, error1);
title("A1 error")
xlabel("Tau")
ylabel("Error")
subplot(2, 2, 2, 'XScale', 'log', 'YScale', 'log')
plot(tau, error2);
title("A2 error")
xlabel("Tau")
ylabel("Error")
subplot(2, 2, 3, 'XScale', 'log', 'YScale', 'log')
plot(tau, error3);
title("A3 error")
xlabel("Tau")
ylabel("Error")
subplot(2, 2, 4, 'XScale', 'log', 'YScale', 'log')
plot(tau, error4);
title("A4 error")
xlabel("Tau")
ylabel("Error")

figure(2)
plot(tau, error4);
title("A4 error")
xlabel("Tau")
ylabel("Error")