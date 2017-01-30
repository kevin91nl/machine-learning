x_min = -2;
x_max = 2;

steps = int64(T / dt);
x = zeros(steps + 1, 1);
v = zeros(steps + 1, 1);
x(1) = x_0;
v(1) = v_0;
t = 0;
for step = 2:steps + 1
    dx = v(step - 1) * dt;
    dxi = sqrt(dt) * normrnd(0, nu);
    if step == 2
        dxi_0 = dxi;
    end
    dv = F(x(step - 1), g) * dt + u(step - 1) * dt + dxi;
    x(step) = x(step - 1) + dx;
    v(step) = v(step - 1) + dv;
    t = t + dt;
end

phi = 0;
if x(step) < x_min | x(step) > x_max
    phi = A;
end

C = phi + 1/2 * sum(u.^2);