clear;clc;
gsp_start;

% sensor_graph = gsp_sensor(10);
% figure;
% gsp_plot_graph(sensor_graph);

% directed 4 taps time series
% time_weight = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0;];
% time_coords = [0 0 0; 0 0 1; 0 0 2; 0 0 3;];
% time_graph = gsp_graph(time_weight, time_coords);
% figure;
% gsp_plot_graph(time_graph);

% undirected time series
% time_graph = gsp_path(3);
% figure;
% gsp_plot_graph(time_graph);

% param.rule = "strong";
% product_graph = gsp_graph_product(sensor_graph,time_graph, param);

% S = sensor_graph.coords;
% T = time_graph.coords(:,1);
% [a,b]=ndgrid(1:size(T, 1),1:size(S, 1));
% product_graph.coords = [S(b,:), T(a,:)];
% figure;
% gsp_plot_graph(product_graph);

sensor_graph_adj = [0 1 0 0; 1 0 1 1; 0 1 0 1;0 1 1 0;];
sensor_graph_coords = [1 2;1 1;0 0; 2 0];
sensor_graph = gsp_graph(sensor_graph_adj, sensor_graph_coords);
figure;
gsp_plot_graph(sensor_graph);

time_graph_adj = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0;];
time_graph_coords = [0 0; 1 0; 2 0; 3 0;];
time_graph = gsp_graph(time_graph_adj, time_graph_coords);
figure;
gsp_plot_graph(time_graph);

product_graph_adj = kron(sensor_graph_adj, time_graph_adj) + kron(sensor_graph_adj, eye(size(time_graph_adj, 1))) + kron(eye(size(sensor_graph_adj, 1)), time_graph_adj);

S = sensor_graph.coords;
T = time_graph.coords(:,1);
[a,b]=ndgrid(1:size(T, 1),1:size(S, 1));
product_graph_coords = [S(b,:), T(a,:)];


product_graph = gsp_graph(product_graph_adj, product_graph_coords);
figure;
gsp_plot_graph(product_graph);

[U, D] = eig(product_graph_adj);
product_graph.U = U;
product_graph.e = diag(D);
