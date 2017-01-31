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