clear;clc;
gsp_start;

% read position matrix
cords = readmatrix('positions.csv');
cords = cords(2:end,2:end);
cords = flip(cords, 2);

% read weight matrix
W = readmatrix('adjacency.csv');
G = gsp_graph(W);
G.coords = cords;

% read graph signal
X = readmatrix("covid_global.csv");
X = X(2:end, 5:9); %306
x = reshape(X', [size(X, 1)*size(X, 2), 1]);
x_max = max(x);
x = x/x_max;

% time series
time_W = eye(5);
time_W = circshift(time_W,1,2);
time_W(end,1) = 0;
time_G = gsp_graph(time_W);

A_strong = kron(W, time_W) + kron(W, eye(size(time_W, 1))) + kron(eye(size(W, 1)), time_W);

N = 1000;
L = 2;
C_pred = rand(1, 7);
lr = 3e-5;
costs = zeros(1, N);
test_error = zeros(1, N);

J = rand(size(x)) > 0.2;
x_sample = x.*J;
[g, cost] = grad(x_sample, C_pred, A_strong, L, J);

for k = 1:N
    [g, cost] = grad(x_sample, C_pred, A_strong, L, J);
    C_pred = C_pred - lr*g;
    costs(k) = cost;
    x_pred = volterra_filter(x_sample, A_strong, L, C_pred);
    test_error(k) = sum((x-x_pred).^2);
    disp([k,cost, test_error(k)]);
end

display("cost = " + cost);
figure;
plot(costs)
title('Training Error')
xlabel("Number of iterations")
ylabel("Error")


figure;
plot(test_error)
title('Test Error')
xlabel("Number of iterations")
ylabel("Error")

% gsp_plot_graph(G);

