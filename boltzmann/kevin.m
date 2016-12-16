% Number of neurons
N = 2;

eta = 0.01;

% Generate the data
P = 2;
w = rand(N, N);
theta = rand(1, N);
samples = 2 * randi([0, 1], P, N) - 1;
samples = [
    1 1
];

% Calculate clamped statistics
stat_1_c = 1 / P * sum(samples);
stat_2_c = 1 / P * samples' * samples;

for t = 1:200
    % Converge to a low energy state and collect samples from it
    num_samples = 500;
    K = 500;
    repr_samples = zeros(num_samples, N);
    s = 2 * randi([0, 1], N, 1) - 1;
    for step = 1:(K + num_samples)
        h = w * s + theta';
        s_flipped = repmat(s', N, 1);
        for i = 1:N
            s_flipped(i, i) = -1 * s(i);
        end
        p_flipping = 1/2 * (1 + tanh(s_flipped * h));
        i = randi([1, N], 1, 1);
        if rand() <= p_flipping
            s(i) = -1 * s(i);
        end

        % Collecting samples
        if step > K
            repr_samples(step - K, :) = s;
        end
    end

    % Compute Z from the representative samples
    Z = 0;
    for i = 1:num_samples
        Z = Z + exp(-E(repr_samples(i, :), w, theta));
    end

    % Calculate statistics
    stat_1 = zeros(1, N);
    stat_2 = zeros(N, N);
    for sample = 1:num_samples
        s = repr_samples(sample, :);
        for i = 1:N
            stat_1(1, i) = stat_1(1, i) + s(i) * p(s, Z, w, theta);
            for j = 1:N
                stat_2(i, j) = stat_2(i, j) + s(i) * s(j) * p(s, Z, w, theta);
            end
        end
    end

    % Calculate derivatives
    dLdtheta = stat_1_c - stat_1;
    dLdw = stat_2_c - stat_2;

    w = w - eta * dLdw;
    theta = theta - eta * dLdtheta;
    t
end

p([-1 -1], Z, w, theta)
p([1 1], Z, w, theta)
p([-1 1], Z, w, theta)

function energy = E(s, w, theta)
    N = numel(s);
    sum = 0;
    energy = dot(theta, s);
    for i = 1:N
        for j = 1:N
            energy = energy + 0.5 * w(i, j) * s(i) * s(j);
        end
    end
end

function probability = p(s, Z, w, theta)
    probability = 1 / Z * exp(-E(s, w, theta));
end