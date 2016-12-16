hold on;
[Z1, w1, theta1, delta_w] = boltzmann([1 1; -1 1], 2, 200, 0.01, 500, 250);
plot(delta_w);
[Z2, w2, theta2, delta_w] = boltzmann([-1 -1], 2, 200, 0.003, 500, 250);
plot(delta_w);
hold off;

legend('1', '2');

p([1 1], Z1, w1, theta1)
p([1 1], Z2, w2, theta2)
p([-1 -1], Z1, w1, theta1)
p([-1 -1], Z2, w2, theta2)

function [Z, w, theta, delta_w] = boltzmann(samples, num_neurons, num_iterations, learning_rate, num_sampled_states, num_burn_in_samples)
    % Generate the data
    w = rand(num_neurons, num_neurons);
    theta = rand(1, num_neurons);

    % Calculate clamped statistics
    P = numel(samples);
    stat_1_c = 1 / P * sum(samples);
    stat_2_c = 1 / P * samples' * samples;

    delta_w = zeros(num_iterations, 1);
    for t = 1:num_iterations
        % Converge to a low energy state and collect samples from it
        repr_samples = generate_samples(num_sampled_states, num_neurons, num_burn_in_samples, w, theta);

        % Compute Z from the representative samples
        Z = 0;
        for i = 1:num_sampled_states
            Z = Z + exp(-E(repr_samples(i, :), w, theta));
        end

        % Calculate statistics
        stat_1 = zeros(1, num_neurons);
        stat_2 = zeros(num_neurons, num_neurons);
        for sample = 1:num_sampled_states
            s = repr_samples(sample, :);
            for i = 1:num_neurons
                stat_1(1, i) = stat_1(1, i) + s(i) * p(s, Z, w, theta);
                for j = 1:num_neurons
                    stat_2(i, j) = stat_2(i, j) + s(i) * s(j) * p(s, Z, w, theta);
                end
            end
        end

        % Calculate derivatives
        dLdtheta = stat_1_c - stat_1;
        dLdw = stat_2_c - stat_2;
        delta_w(t) = learning_rate * sum(sum(abs(dLdw)));

        w = w - learning_rate * dLdw;
        theta = theta - learning_rate * dLdtheta;
        t
    end
end

function repr_samples = generate_samples(num_sampled_states, num_neurons, num_burn_in_samples, w, theta)
    repr_samples = zeros(num_sampled_states, num_neurons);
    s = 2 * randi([0, 1], num_neurons, 1) - 1;
    for step = 1:(num_burn_in_samples + num_sampled_states)
        h = w * s + theta';
        s_flipped = repmat(s', num_neurons, 1);
        for i = 1:num_neurons
            s_flipped(i, i) = -1 * s(i);
        end
        p_flipping = 1/2 * (1 + tanh(s_flipped * h));
        i = randi([1, num_neurons], 1, 1);
        if rand() <= p_flipping
            s(i) = -1 * s(i);
        end

        % Collecting samples
        if step > num_burn_in_samples
            repr_samples(step - num_burn_in_samples, :) = s;
        end
    end
end

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