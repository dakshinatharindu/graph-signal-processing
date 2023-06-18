clear;clc;
gsp_start;

sensor_graph_adj = [0 1 0 0; 1 0 1 1; 0 1 0 1;0 1 1 0;];
sensor_graph_coords = [1 2;1 1;0 0; 2 0];
sensor_graph = gsp_graph(sensor_graph_adj, sensor_graph_coords);
figure;
gsp_plot_graph(sensor_graph);
title("Sensor Graph")

time_graph_adj = [0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0;];
time_graph_coords = [0 0; 1 0; 2 0; 3 0;];
time_graph = gsp_graph(time_graph_adj, time_graph_coords);
figure;
gsp_plot_graph(time_graph);
title("Time Graph")

A_strong = kron(sensor_graph_adj, time_graph_adj) + kron(sensor_graph_adj, eye(size(time_graph_adj, 1))) + kron(eye(size(sensor_graph_adj, 1)), time_graph_adj);

S = sensor_graph.coords;
T = time_graph.coords(:,1);
[a,b]=ndgrid(1:size(T, 1),1:size(S, 1));
product_graph_coords = [S(b,:), T(a,:)];

product_graph = gsp_graph(A_strong, product_graph_coords);
figure;
gsp_plot_graph(product_graph);
title("Product Graph")

% eigen-decomposistion
% [U, D] = eig(A_strong);
% product_graph.U = U;
% product_graph.e = diag(D);

% laplacian
product_graph = gsp_compute_fourier_basis(product_graph);
A_strong = product_graph.L;

x = randi([0 20], size(A_strong, 1), 1);
X = gsp_gft(product_graph, x);

%%%%%%%%%%%%%%%% Linear filter design%%%%%%%%%%%%%%%%%%%%%%%%%
filter_taps = [3 2 1 2];
h = zeros(size(A_strong));
H = zeros(size(A_strong));
for i = 1:length(filter_taps)
    h = h + filter_taps(i) * A_strong^(i-1);
    H = H + filter_taps(i) * diag(product_graph.e)^(i-1);
end
y_linear = h * x;
Y = H * X;

figure;
gsp_plot_signal_spectral(product_graph, X);
title("Input Spectrum")

y_linear_hat = gsp_gft(product_graph, y_linear);
figure;
gsp_plot_signal(product_graph, x);
title("Input Signal")

%%%%%%%%%%%%%%%% Volterra filter design %%%%%%%%%%%%%%%%%%%%%%%%%

c0 = 1;
c1 = [2 3 4 5];
c2 = [[1 2 3 4]; [3 5 6 7]; [4 2 6 7]; [8 3 1 2]];

filter_order = 2;
num_taps = 4;
% filter_coefs = [c0 c1 c2];

temp = zeros(size(x));
output = zeros(size(x));

sz = size(x);
shifted_signals = zeros(sz(1), num_taps);

for i = 1:num_taps
    shifted_signals(:, i) = A_strong^(i-1) * x;
end
% iterating over taps

% zeroth order term

for i = 1:length(x)
    temp(i) = c0 ;
end

output = output + temp;

% first order term
temp = zeros(size(x));
for i = 1:num_taps
    temp = temp + c1(i) * shifted_signals(:,i);
end

output = output + temp;

% second order term

temp = zeros(size(x));
for i = 1:num_taps
    for j = i:num_taps
        temp = temp + c2(i, j) * shifted_signals(:,i) .* shifted_signals(:,j);
    end
end

output = output + temp;

figure;
gsp_plot_signal(product_graph, output);
title("Non-linear output Signal")

output_spectrum = gsp_gft(product_graph, output);
figure;
gsp_plot_signal_spectral(product_graph, output_spectrum);
title("Output Spectrum")


