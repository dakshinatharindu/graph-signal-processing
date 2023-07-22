clear;clc;
% gsp_start;


% read position matrix
cords = dlmread("../volterra-experiments/matlab_code/positions.csv");
cords = cords(1:end,2:end);
cords = flip(cords, 2);

% read weight matrix
W = dlmread('../volterra-experiments/matlab_code/adjacency.csv');

% variables
num_time_steps = 10;
num_taps = 2;
sampling_probability = 0.3;

X = csvread("../volterra-experiments/matlab_code/covid_global.csv",1, 4);
X = X(1:end, 1:num_time_steps); %306
x = reshape(X', [size(X, 1)*size(X, 2), 1]);
x = max(x, 0);
x_max = max(x);
x = x/x_max;

% time series
time_W = eye(num_time_steps);
time_W = circshift(time_W,1,2);
time_W(end,1) = 0;

A_strong = kron(W, time_W) + kron(W, eye(size(time_W, 1))) + kron(eye(size(W, 1)), time_W);


rng('default');
J = rand(size(x)) > sampling_probability;
x_sample = x.*J;
rng('shuffle')

X_unrolled = [zeros(size(x_sample))];
weight_mu = [1];

for i = 1:num_taps+1
    X_unrolled = [X_unrolled, A_strong^(i-1)*x_sample];
    weight_mu = [weight_mu, 1];
end

for k1 = 1:num_taps+1
    for k2 = 1:num_taps+1
        X_unrolled = [X_unrolled, A_strong^(k1-1)*x_sample.*A_strong^(k2-1)*x_sample];
        weight_mu = [weight_mu, 1/100];
    end
end

for k1 = 1:num_taps+1
    for k2 = 1:num_taps+1
        for k3 = 1:num_taps+1
            X_unrolled = [X_unrolled, A_strong^(k1-1)*x_sample.*A_strong^(k2-1)*x_sample.*A_strong^(k3-1)*x_sample];
            weight_mu = [weight_mu, 1/1000];
        end
    end
end

x_unrolled_max = max(max(X_unrolled));
% X_unrolled = X_unrolled/x_unrolled_max;

H = abs(rand(size(X_unrolled, 2),1))/100;
error = zeros(size(x));


% hyperparameters
% mu = 3e-5;
epochs = 5e6;
mu_array = [7e-4, 2e-3, 7e-3, 7e-3, 7e-3]*0.0001;
lambda = 1e-8;
format short g;
% gradient descent
for i = 1:epochs
    error = x_sample - (X_unrolled*H).*J;
    mu = mu_array(ceil((i/epochs)*length(mu_array)));
    prev_H2_1 = H(2);
    H = H + mu*(weight_mu').*(X_unrolled/x_unrolled_max)'*error;
    prev_H2_2 = H(2);
    H(2) = H(2) - lambda*2*abs(H(2))*(-1)^(H(2)<0);
    
    if (rem(i,1000)==0)
        disp([i, norm(error), norm(x-X_unrolled*H), prev_H2_2-prev_H2_1, H(2), lambda*2*abs(H(2))*(-1)^(H(2)<0), mu]);
    end
    
end
disp(norm(x-x_sample));

figure;
stem(x);
hold on;
stem(X_unrolled*H);
legend('original', 'reconstructed');