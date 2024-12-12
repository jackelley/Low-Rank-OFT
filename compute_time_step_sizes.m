function [times, n, a, b] = compute_time_step_sizes(dt_initial, t_end, dt_end)
% Given an initial step size and some end time and step size at that time,
% return an array of time step sizes along with the the number of time 
% steps n and the constants a and b.


% Find constants
R = (dt_end / dt_initial) - 1;
a = t_end / R;
b = 1 + (R * (dt_initial / t_end));

% Fill times array
times = zeros(0, round(t_end / (a * b)));
times(2) = dt_initial;
current_time = times(2);
n = 1;
while current_time <= t_end
    times(n + 1) = a * (b^n - 1);
    current_time = times(n + 1);
    n = n + 1;
end

% Truncate unused space
times = times(:, 1:n);

end
