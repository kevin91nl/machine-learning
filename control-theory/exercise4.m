hold on;
for r = 1:100
    nu = 0.1;
    T = 5;
    dt = 0.1;
    t = 0;
    steps = T / dt;

    u = zeros(steps + 1, 1);
    x = zeros(steps + 1, 1);
    x(1) = 0;
    for step = 2:(steps + 1)
        u(step) = tanh(x(step - 1) / (nu * (T - t)) - x(step - 1)) / (T - t);
        dxi = sqrt(dt) * normrnd(0, nu);
        x(step) = u(step) * dt + dxi;
        t = t + dt;
    end

    plot(0:dt:T, x);
end
hold off;