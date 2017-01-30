function out = Control(assignment, action, varargin)
    % Control  Script for performing tasks related to the Control theory
    % lectures. The first argument specifies the assignment to run. The
    % seconds argument specifies which action to run.
    %
    % The options for the first argument are the following:
    %   'crw': Controlled Random Walk
    %   'mcp': Mountain Car Problem
    %
    % The following global options are optional:
    %    seed  The seed for the random number generator to use 
    %          (default: -1 which will not use any seed).
    %    path  The path for the output (default: 'output/').
    %
    % The following script calls are possible:
    %
    %   Control('crw', 'simulate')
    %     Run a simulation and store the result in the given output file.
    %     Options (optional):
    %       noise         The noise level.
    %       horizon       The end-time.
    %       dt            Change in time per step.
    %       trials        Number of trials.
    %       show_state    When true, a plot of the states will be shown.
    %       show_control  When true, a plot of the optimal control will be
    %                     shown.
    %     Example calls:
    %       Control('crw', 'simulate');
    %       Control('crw', 'simulate', 'noise', 0.1, 'horizon', 1, 'dt', 0.01, 'trials', 1, 'show_state', true, 'show_control', true, 'seed', 5);
    %
    %   Control('crw', 'explore')
    %     Options (optional):
    %       noise_min         Minimum noise level.
    %       noise_stepsize    Noise step size.
    %       noise_max         Maximum noise level.
    %       horizon_min       Minimum horizon.
    %       horizon_stepsize  Horizon step size.
    %       horizon_max       Maximum horizon.
    %       dt                Change in time per step.
    %       trials            Number of trials.
    %
    %     Example calls:
    %       Control('crw', 'explore', 'noise_min', 1.0, 'noise_max', 10.0, 'noise_stepsize', 1.0, 'trials', 100);
    %       Control('crw', 'explore', 'noise_min', 1.0, 'noise_max', 15.0, 'noise_stepsize', 2.0, 'trials', 1000, 'seed', 5);
    %       Control('crw', 'explore', 'noise_min', 1.0, 'noise_max', 10.0, 'noise_stepsize', 1.0, 'trials', 1000, 'horizon_min', 1, 'horizon_stepsize', 10, 'horizon_max', 31, 'seed', 5);
    
    % Check arguments and global options
    p = inputParser;
    p.KeepUnmatched = true;
    addRequired(p, 'assignment', @(x) any(validatestring(x, {'crw', 'mcp'})));
    addRequired(p, 'action', @ischar);
    addParameter(p, 'seed', -1, @isnumeric);
    addParameter(p, 'path', 'output/', @ischar);
    parse(p, assignment, action, varargin{:});
    
    % Set the seed for the RNG (if not -1)
    if p.Results.seed ~= -1
        rng(p.Results.seed);
    end
    
    % Create the output directory if it does not exist
    path = p.Results.path;
    if exist(path, 'dir') ~= 7
        mkdir(path);
    end
    
    % Execute code for the Controlled Random Walk assignment
    if strcmp(p.Results.assignment, 'crw')
        CRW(p.Results.action, path, varargin{:});
    elseif strcmp(p.Results.assignment, 'mcp')
        
    end
end

%% Mountain Car problem
function out = MCP(action, path, varargin)

end

%% Controlled Random Walk problem
function out = CRW(action, path, varargin)
    % Check arguments
    p = inputParser;
    p.KeepUnmatched = true;
    addRequired(p, 'action', @(x) any(validatestring(x, {'simulate', 'explore'})));
    parse(p, action);

    % Run the simulation action
    if strcmp(action, 'simulate')
        CRW_simulate(path, varargin{:});
    end
    
    if strcmp(action, 'explore')
        CRW_explore(path, varargin{:});
    end
end

function out = CRW_simulate(path, varargin)
    % Parse the options
    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'noise', 0.1, @isnumeric);
    addParameter(p, 'seed', -1, @isnumeric);
    addParameter(p, 'horizon', 1, @isnumeric);
    addParameter(p, 'dt', 0.01, @isnumeric);
    addParameter(p, 'trials', 1, @isnumeric);
    addParameter(p, 'show_state', true, @islogical);
    addParameter(p, 'show_control', true, @islogical);
    parse(p, varargin{:});
    
    % Store the options in variables
    noise = p.Results.noise;
    seed = p.Results.seed;
    horizon = p.Results.horizon;
    dt = p.Results.dt;
    trials = p.Results.trials;
    show_state = p.Results.show_state;
    show_control = p.Results.show_control;
    
    % Run the simulations
    steps = int32(horizon / dt) + 1;
    states = zeros(trials, steps);
    controls = zeros(trials, steps);
    for trial = 1:trials
        [x, t, steps, xi, u_star] = crw_simulate_optimal_control(noise, horizon, dt);
        states(trial, :) = x;
        controls(trial, :) = u_star;
    end

    % Make a figure
    figure('Visible', 'off');

    % Plot the state during the simulation
    if show_state && show_control
        subplot(2, 1, 1);
    end
    if show_state
        plot(t, states);
        title('Simulation');
        xlabel('Time');
        ylabel('State');
        axis([min(t), max(t), -1.5, 1.5]);
    end

    % Plot the optimal control
    if show_state && show_control
        subplot(2, 1, 2);
    end
    if show_control
        plot(t, controls);
        title('Optimal Control');
        xlabel('Time');
        ylabel('Control');
        axis([min(t), max(t), -1.5, 1.5]);
    end

    % Save the figure
    filename = strcat(path, 'crw_simulate');
    filename = strcat(filename, '_noise=', num2str(noise));
    filename = strcat(filename, '_horizon=', num2str(horizon));
    filename = strcat(filename, '_dt=', num2str(dt));
    filename = strcat(filename, '_trials=', num2str(trials));
    filename = strcat(filename, '_show-state=', num2str(show_state));
    filename = strcat(filename, '_show-control=', num2str(show_control));
    filename = strcat(filename, '_seed=', num2str(seed));
    filename = strcat(filename, '.png');
    saveas(gcf, filename);
end

function out = CRW_explore(path, varargin)
    % Parse the options
    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'noise_min', 0.1);
    addParameter(p, 'noise_max', 2.0);
    addParameter(p, 'noise_stepsize', 0.1);
    addParameter(p, 'horizon_min', 1.0);
    addParameter(p, 'horizon_max', 5.0);
    addParameter(p, 'horizon_stepsize', 2.0);
    addParameter(p, 'seed', -1, @isnumeric);
    addParameter(p, 'dt', 0.01, @isnumeric);
    addParameter(p, 'trials', 1, @isnumeric);
    parse(p, varargin{:});
    
    % Store the options in variables
    noise_min = p.Results.noise_min;
    noise_max = p.Results.noise_max;
    noise_stepsize = p.Results.noise_stepsize;
    horizon_min = p.Results.horizon_min;
    horizon_max = p.Results.horizon_max;
    horizon_stepsize = p.Results.horizon_stepsize;
    seed = p.Results.seed;
    dt = p.Results.dt;
    trials = p.Results.trials;
    
    % Calculate the noise and horizon
    noise_steps = floor((noise_max - noise_min) / noise_stepsize) + 1;
    horizon_steps = floor((horizon_max - horizon_min) / horizon_stepsize) + 1;
    horizon_data = zeros(horizon_steps, 1);
    
    % Create the figure
    figure('Visible', 'off');
    
    % Create the plots
    hold on;
    for horizon_step = 1:horizon_steps
        % Initialize the noise, horizon and error
        noise_data = zeros(noise_steps, 1);
        error_data = zeros(noise_steps, 1);
        horizon = horizon_min + (horizon_step - 1) * horizon_stepsize;
        horizon
        horizon_data(horizon_step, 1) = horizon;
        
        % Loop through all noise levels
        for noise_step = 1:noise_steps
            noise = double(noise_min + (noise_step - 1) * noise_stepsize);
            noise
            error = 1;
            
            % Loop through all trials
            for trial = 1:trials
                [x, t, steps, xi, u_star] = crw_simulate_optimal_control(noise, horizon, dt);
                error = error + log(0.5 * normpdf(x(steps), -1, 0.5) + 0.5 * normpdf(x(steps), 1, 0.5));
            end
            
            % Update the error and noise data
            error_data(noise_step, 1) = error / double(noise_steps);
            noise_data(noise_step, 1) = noise;
        end
        % Create the plot
        plot(noise_data(:, 1), error_data(:, 1), 'DisplayName', strcat('T = ', num2str(horizon)));
    end
    hold off;
    
    % Update plot attributes
    legend(gca, 'show', 'Location', 'NorthWest');
    title('Explore parameter space');
    xlabel('\nu');
    ylabel('Error');
    
    % Save the figure
    filename = strcat(path, 'crw_explore');
    filename = strcat(filename, '_noise-min=', num2str(noise_min));
    filename = strcat(filename, '_noise-stepsize=', num2str(noise_stepsize));
    filename = strcat(filename, '_noise-max=', num2str(noise_max));
    filename = strcat(filename, '_horizon-min=', num2str(horizon_min));
    filename = strcat(filename, '_horizon-stepsize=', num2str(horizon_stepsize));
    filename = strcat(filename, '_horizon-max=', num2str(horizon_max));
    filename = strcat(filename, '_dt=', num2str(dt));
    filename = strcat(filename, '_trials=', num2str(trials));
    filename = strcat(filename, '_seed=', num2str(seed));
    filename = strcat(filename, '.png');
    saveas(gcf, filename);
end

function [x, t, steps, noise, u_star] = crw_simulate_optimal_control(nu, T, dt)
    % CRW_SIMULATE  Simulate optimal control in the Control Random Walk
    % problem with noise parameter NU and horizon T with change in time DT.
    % It returns the states x(t=0), ..., x(t=T), the times t=0, ..., t=T, 
    % the number of simulation steps, the noise from t=0 to t=T and the 
    % optimal control u*(t=0), ..., u*(t=T).

    % Compute the number of steps
    steps = int64(T / dt) + 1;

    % Initialize the variables
    x = zeros(steps, 1);
    noise = zeros(steps, 1);
    u_star = zeros(steps, 1);
    t = zeros(steps, 1);

    % Perform the simulation
    for step = 2:steps
        % Compute the optimal control
        u_star(step) = crw_optimal_control(x(step - 1), t(step - 1), nu, T);

        % Compute the noise
        dxi = sqrt(dt) * normrnd(0, nu);

        % Compute the change in state
        dx = u_star(step) * dt + dxi;

        % Update the state
        x(step) = x(step - 1) + dx;

        % Update the time
        t(step) = t(step - 1) + dt;
    end
end

function u_star = crw_optimal_control(x, t, nu, T)
    % CRW_OPTIMAL_CONTROL  Compute the optimal control for the Controlled 
    % Random Walk problem at state X and time T for noise parameter NU and 
    % horizon T.
    u_star = (tanh(x / (nu * (T - t))) - x) / (T - t);
end