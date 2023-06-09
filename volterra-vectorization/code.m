clear;clc;
gsp_start;

% read position matrix
cords = dlmread("../volterra-experiments/matlab_code/positions.csv");
cords = cords(1:end,2:end);
cords = flip(cords, 2);

% read weight matrix
W = dlmread('../volterra-experiments/matlab_code/adjacency.csv');

% variables
num_time_steps = 5;
num_taps = 2;
sampling_probability = 0.3;

X = csvread("../volterra-experiments/matlab_code/covid_global.csv",1, 4);
X = X(1:end, 1:num_time_steps); %306
x = reshape(X', [size(X, 1)*size(X, 2), 1]);
x_max = max(x);
x = x/x_max;

% time series
time_W = eye(num_time_steps);
time_W = circshift(time_W,1,2);
time_W(end,1) = 0;

A_strong = kron(W, time_W) + kron(W, eye(size(time_W, 1))) + kron(eye(size(W, 1)), time_W);

J = rand(size(x)) > sampling_probability;
x_sample = x.*J;

X_unrolled = [ones(size(x_sample))];

for i = 1:num_taps+1
    X_unrolled = [X_unrolled, A_strong^(i-1)*x_sample];
end

for k1 = 1:num_taps+1
    for k2 = 1:num_taps+1
        X_unrolled = [X_unrolled, A_strong^(k1-1)*x_sample.*A_strong^(k2-1)*x_sample];
    end
end

% for k1 = 1:num_taps+1
%     for k2 = 1:num_taps+1
%         for k3 = 1:num_taps+1
%             X_unrolled = [X_unrolled, A_strong^(k1-1)*x_sample.*A_strong^(k2-1)*x_sample.*A_strong^(k3-1)*x_sample];
%         end
%     end
% end

x_unrolled_max = max(max(X_unrolled));
% X_unrolled = X_unrolled/x_unrolled_max;

H = rand(size(X_unrolled, 2),1);
error = zeros(size(x));

% hyperparameters
mu = 3e-5;
epochs = 100000;

% gradient descent
for i = 1:epochs
    error = x - X_unrolled*H;
    H = H + mu*(X_unrolled/x_unrolled_max)'*error;
    disp(norm(error));
end

figure;
plot(x);
hold on;
plot(X_unrolled*H);
legend('original', 'reconstructed');

