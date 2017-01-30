% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal

METHOD='sa';
NEIGHBORHOODSIZE=3;
n_restart = 1;

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
                x_new1 = x;
                x_new2 = x;
                x_new1(bit1) = x_new1(bit1) * -1;
                x_new2(bit2) = x_new2(bit2) * -1;
                E_new1 = E(x_new1,w);
                E_new2 = E(x_new2,w);
                if E_new1 < E1 & E_new1 < E_new2
                    x = x_new1;
                    E1 = E_new1;
                    flag = 1;
                end
                if E_new2 < E1 & E_new2 < E_new1
                    x = x_new2;
                    E1 = E_new2;
                    flag = 1;
                    break;
                end
            otherwise,
                bits = randperm(numel(x), NEIGHBORHOODSIZE);
                x_old = x;
                x_old
                for index = 1:numel(bits)
                    x_new = x_old;
                    bit = bits(index);
                    x_new(bit) = x_new(bit) * -1;
                    bit
                    x_new
                    E_new = E(x_new,w);
                    E_new
                    if E_new < E1
                        E_new
                        x = x_new;
                        E1 = E_new;
                        flag = 1;
                        break;
                    end
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
	switch NEIGHBORHOODSIZE,
        case 1,
			% estimate maximum dE in single spin flip
        case 2,
			% estimate maximum dE in pair spin flip
        end;
	beta_init=1/max_dE;	% sets initial temperature
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain

	beta=beta_init;
	E_bar(1)=1;
	t2=1;
	while E_bar(t2) > 0,
		t2=t2+1;
		beta=beta*factor;
		E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one random bit i
				% perform Metropolis Hasting step
			case 2,
				% choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
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
plot(1:t2,E_outer(1:t2),1:t2,E_bar(1:t2))

