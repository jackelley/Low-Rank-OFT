clear; clc; close all
n = 2^4;
L = 1;
h = L / n;
x = h * (0:n-1)';
a = exp(cos((2 * pi / L) * x));
% Frequency vector
k = [0:n/2-1, -n/2:-1]' * (2*pi/L); % Adjusted frequency vector
% Fourier transform of the derivative
tf_a = 1i .* k .* fft(a);
da = ifft(tf_a, 'symmetric'); % Use 'symmetric' to avoid complex numerical errors
% Plotting
% plot(x, da, 'b', x, gradient(a) ./ gradient(x), 'g')

f = a;
f_hat = fft(a);
u_hat = (1 - (k.^2)).^(-1/2) .* f_hat;
u = ifft(u_hat, 'symmetric');

plot(x, u)