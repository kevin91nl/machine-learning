L = @(x) -1 - 1/2 * (tanh(2*x + 2) - tanh(2*x - 2));
dLdx = @(x) sech(2*x + 2).^2 - sech(2*x - 2).^2;
F = @(x, g) -g * dLdx(x) / sqrt(1 + dLdx(x).^2);

nu = 0.3;
dt = 0.1;
T = 20;
steps = T / dt;
dxi = normrnd(0, sqrt(nu * dt), steps, 1);

s = zeros(steps + 1, 2);
s(1, :) = [0 0];

u = zeros(steps, 1);
M = [0 dt 0 0 ; dt 0 dt 1];
for step = 1:steps
    ds = M * [F(s(step, 1), -9.81) s(step, 2) u(step) dxi(step)]';
    s(step + 1, :) = s(step, :) + ds';
    
    clf;
    hold on;
    plot(-2:0.1:2, L(-2:0.1:2));
    plot(s(step, 1), L(s(step, 1)), 'r.', 'MarkerSize', 20);
    xlim([-2 2]);
    hold off;
    drawnow;
    pause(0.01);
end