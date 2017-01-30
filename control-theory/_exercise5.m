L = @(x) -1 -1/2*(tanh(2*x + 2) - tanh(2*x - 2));
dLdx = @(x) sech(2 - 2*x).^2 - sech(2*x + 2).^2;
F = @(x, g) -g * dLdx(x) ./ sqrt(1 + dLdx(x).^2);

dt = 0.01;
T = 1;
steps = int64(T / dt);
u = -ones(steps, 1);
u = u * 0;

rng(5);
% 
% N = 50;
% 
% dx = 0.1;
% x_min = -2;
% x_max = 2;
% num_x = int64((x_max - x_min) / dx) + 1;
% 
% dv = 0.1;
% v_min = -1;
% v_max = 1;
% num_v = int64((v_max - v_min) / dv) + 1;
% 
% M = zeros(num_x, num_v);
% for step_x = 1:num_x
%     for step_v = 1:num_v
%         [step_x, step_v]
%         x = x_min + (step_x - 1) * dx;
%         v = v_min + (step_v - 1) * dv;
%         M(step_x, step_v) = estimate_optimal_cost_to_go(N, x, v, zeros(steps, 1), -1, 1, T, 0, dt, L, dLdx, F);
%         [x v M(step_x, step_v)];
%     end
% end
% [X V] = meshgrid(x_min:dx:x_max, v_min:dv:v_max);
% surf(X', V', M);

[x, v, phi, C, dxi_0] = run_simulation(-1, 1, zeros(steps, 1), -1, 1, T, 0, dt, L, dLdx, F);
for step = 1:numel(x)
    clf;
    domain = -2:0.01:2;
    hold on;
    plot(domain, L(domain), 'b');
    plot(x(step), L(x(step)), '.r', 'MarkerSize', 20);
    hold off;
    drawnow;
    pause(0.01);
end

function J = estimate_optimal_cost_to_go(N, x_0, v_0, u, A, g, T, nu, dt, L, dLdx, F)
    S = 0;
    for n = 1:N
        [x, v, phi, C, dxi_0] = run_simulation(x_0, v_0, u, A, g, T, nu, dt, L, dLdx, F);
        S = S + exp(-phi);
    end
    J = -log(S / N);
end

function [x, v, phi, C, dxi_0] = run_simulation(x_0, v_0, u, A, g, T, nu, dt, L, dLdx, F)
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
end