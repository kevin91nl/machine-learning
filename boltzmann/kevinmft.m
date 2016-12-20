% Set the seed
rng(5);

% Set the number of neurons
num_neurons = 784;

% Initialize the weights and biases
w = rand(num_neurons, num_neurons);
theta = rand(num_neurons, 1);
m = ones(num_neurons, 1);

% Set the diagonal of the weights to zero (no self-loops)
w(logical(eye(size(w)))) = 0;

s = magic(num_neurons);
for i = 1:100
    m = tanh(w * m - theta);
end

calculate_log_q(ones(num_neurons, 1), m);
samples = generate_samples(20, m);

function log_q = calculate_log_q(s, m)
    log_q = sum(calculate_log_qi(s, m));
end

function log_probability = calculate_log_qi(s, m)
    log_probability = log(1 + s .* m') + numel(s) * log(1/2);
end

function samples = generate_samples(num_samples, m)
    samples = 1;
    num_neurons = numel(m);
    exp(calculate_log_q(ones(num_neurons, 1), m))
end