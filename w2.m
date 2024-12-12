function [w2] = w2(a, b)

function [c] = C(x)
% Finds non-normalized fresnel cosine
A = sqrt(2/pi);
c = 1/A * fresnelc(A * x);
end

function [s] = S(x)
% Finds non-normalized fresnel sine
A = sqrt(2/pi);
s = 1/A * fresnels(A * x);
end

C_a = C(sqrt(a));
C_b = C(sqrt(b));
S_a = S(sqrt(a));
S_b = S(sqrt(b));
w2 = ((1 + 1i) ./ (sqrt(2 * pi) * (b - a))) .* (sqrt(a) .* exp(1i * a) ...
    - sqrt(b) .* exp(1i * b) + ((1 + 2i * a) .* (C_b - C_a + (1i * S_b) - (1i * S_a))));
end