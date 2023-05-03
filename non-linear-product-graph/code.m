clear;clc;

sensor_graph = gsp_sensor(20);
figure;
gsp_plot_graph(sensor_graph);

% directed 4 taps time series
% time_weight = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0;];
% time_coords = [0 0 0; 0 0 1; 0 0 2; 0 0 3;];
% time_graph = gsp_graph(time_weight, time_coords);
% figure;
% gsp_plot_graph(time_graph);

% undirected time series
time_graph = gsp_path(5);
figure;
gsp_plot_graph(time_graph);

param.rule = "cartesian";
product_graph = gsp_graph_product(sensor_graph,time_graph, param);

S = sensor_graph.coords;
T = time_graph.coords(:,1);
[a,b]=ndgrid(1:size(T, 1),1:size(S, 1));
product_graph.coords = [S(b,:), T(a,:)];
figure;
gsp_plot_graph(product_graph);