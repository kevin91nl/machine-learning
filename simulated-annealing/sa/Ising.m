% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal

METHOD='sa';
NEIGHBORHOODSIZE=1;
n_restart =100;

switch METHOD,
case 'iter'

	E_min = 10000;
    x = 2*(rand(1,n)>0.5)-1;
	for t=1:n_restart,

		% initialize
		E1 = E(x,w);
		flag = 1;
	
		while flag == 1,
			flag = 0;
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one bit i
				% compute dE directly instead of subtracting E's of
				% different states because of efficiency
                bit = randi([1, numel(x)]);
                x_new = x;
                x_new(bit) = x_new(bit) * -1;
                E2 = E(x_new,w);
                % Check whether the new energy is smaller than the old
                % energy and update the 'best' state
                if E2 < E1
                    x = x_new;
                    E1 = E2;
                    flag = 1;
                    break;
                end
			case 2,
				% choose new x by flipping bits i,j
                bit1 = randi([1, numel(x)]);
                bit2 = randi([1, numel(x)]);
                while bit2 == bit1
                    bit2 = randi([1, numel(x)]);
                end
                % Make 3 states and check whether the energy decreases
                x_new1 = x;
                x_new2 = x;
                x_new3 = x;
                x_new1(bit1) = x_new1(bit1) * -1;
                x_new2(bit2) = x_new2(bit2) * -1;
                x_new3(bit1) = x_new3(bit1) * -1;
                x_new3(bit2) = x_new3(bit2) * -1;
                E_new1 = E(x_new1,w);
                E_new2 = E(x_new2,w);
                E_new3 = E(x_new3,w);
                
                % Check if the energy of the state in which the first bit
                % is flipped is the best neighbour in terms of energy
                if E_new1 < E1 & E_new1 < E_new2 & E_new1 < E_new3
                    x = x_new1;
                    E1 = E_new1;
                    flag = 1;
                end
                
                % Check if the energy of the state in which the second bit
                % is flipped is the best neighbour in terms of energy
                if E_new2 < E1 & E_new2 < E_new1 & E_new2 < E_new3
                    x = x_new2;
                    E1 = E_new2;
                    flag = 1;
                    break;
                end
                
                % Check if the energy of the state in which both bits are
                % flipped is the best neighbour in terms of energy
                if E_new3 < E1 & E_new3 < E_new1 & E_new3 < E_new2
                    x = x_new3;
                    E1 = E_new3;
                    flag = 1;
                    break;
                end
            end;
		end;
		E_min = min(E_min,E1);
	end;
	E_min
case 'sa'
	% initialize
	x = 2*(rand(1,n)>0.5)-1;
	E1 = E(x,w);
	E_outer=zeros(1,100);	%stores mean energy at each temperature
	E_bar=zeros(1,100);		% stores std energy at each temperature

	% initialize temperature
	max_dE=0;
    x_old = x;
    
    % Estimate the maximum change of energy by taking random samples and
    % take one random neighbour and compare the energies
	switch NEIGHBORHOODSIZE,
        case 1,
			% estimate maximum dE in single spin flip
            for round = 1:10000
                x_1 = 2*(rand(1,n)>0.5)-1;
                bit = randi(1, n);
                x_2 = x_1;
                x_2(bit) = x_2(bit) * -1;
                dE = E(x_2, w) - E(x_1, w);
                max_dE = max(max_dE, dE);
            end
        case 2,
			% estimate maximum dE in pair spin flip
        end;
        
	beta_init=1/max_dE;	% sets initial temperature
    
    %beta_init = 0.1;
    
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain
    
    beta_init = 2;
    factor = 2;
    T_1=1000;
    
	beta=beta_init;
	E_bar(1)=1;
	t2=0;
	while t2 == 0 | E_bar(t2) > 0,
        t2=t2+1;
		beta=beta*factor;
		E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one random bit i
				% perform Metropolis Hasting step
                bit = randi(n, 1);
                x_new = x;
                x_new(bit) = x_new(bit) * -1;
                E2 = E(x_new,w);
                dE = E2 - E1;
                a = exp(-dE * beta);
                a = min(1, a);
                if a > rand()
                    x = x_new;
                    E1 = E2;
                end
                E_all(1,t1) = E1;
			case 2,
				% choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
                bit1 = randi(n, 1);
                bit2 = bit1;
                
                % Choose a second bit which is unequal to the first bit
                while bit2 == bit1
                    bit2 = randi(n, 1);
                end
                
                % Compute the proposed states
                x_new = x;
                x_new(bit1) = x_new(bit1) * -1;
                x_new(bit2) = x_new(bit2) * -1;
                
                % First check for the state in which both bits are flipped
                E2 = E(x_new, w);
                dE = E2 - E1;
                a = exp(-dE * beta);
                a = min(1, a);
                if a > rand()
                    x = x_new;
                    E1 = E2;
                else
                    % Then check for the state in which only the first bit is
                    % flipped
                    x_new = x;
                    x_new(bit1) = x_new(bit1) * -1;
                    E2 = E(x_new, w);
                    dE = E2 - E1;
                    a = exp(-dE * beta);
                    a = min(1, a);
                    if a > rand()
                        x = x_new;
                        E1 = E2;
                    else
                        % Finally check for the state in which only the
                        % second bit is flipped
                        x_new = x;
                        x_new(bit1) = x_new(bit1) * -1;
                        E2 = E(x_new, w);
                        dE = E2 - E1;
                        a = exp(-dE * beta);
                        a = min(1, a);
                        if a > rand()
                            x = x_new;
                            E1 = E2;
                        end
                    end
                end
                E_all(1,t1) = E1;
			end;
			% E1 is energy of new state
			E_all(t1)=E1;
		end;
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
	end;
	E_min=E_all(1) % minimal energy 
end;

subplot(1, 2, 1);
plot(1:t2,E_outer(1:t2))
xlabel('Iterations');
ylabel('Energy');
title('Mean energy');

subplot(1, 2, 2);
plot(1:t2,E_bar(1:t2))
xlabel('Iterations');
ylabel('Energy');
title('Energy standard deviation');

