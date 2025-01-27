function [U, S, V] = truncsum(tolerance, U_hat, S_hat, V_hat)
%TRUNCSUM given input matrices, compute the truncated SVD of their sum

% column pivoted QR
[QU, RU, PU] = qr(U_hat);
[QV, RV, PV] = qr(V_hat);
[U, S, V] = svd(RU * PU * S_hat * PV' * RV');

% find tolerace value
sv = diag(S);
sv_sum_squared = 0;
j = length(sv);
while sqrt(sv_sum_squared) <= tolerance
    sv_sum_squared = sv_sum_squared + real(sv(j))^2;
    j = j - 1;

end
j = j + 1;

U = U(:, 1:j);
S = S(1:j, 1:j);
V = V(:, 1:j);

U = QU * U;
V = QV * V;
