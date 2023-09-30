clear all, close all, clc;
%%
load('C:\Users\daksh\Projects\GSP\GraphTRSS\covid_19_experiment_global\graph_construction\full_graph.mat');
load('C:\Users\daksh\Projects\GSP\GraphTRSS\covid_19_experiment_global\covid_19_new_cases.mat');
load('C:\Users\daksh\Projects\GSP\GraphTRSS\covid_19_experiment_global\sampling_matrix_path\sample_matrices.mat');
x_matrix = Data;
x_matrix = x_matrix(:, 100:103);
x_matrix_max = max(max(x_matrix));
x_matrix = x_matrix/x_matrix_max;
%%
time_W = eye(size(x_matrix, 2));
time_W = circshift(time_W,1,2);
time_W(end,1) = 0;
%%
A = kron(G.W, eye(size(time_W, 1))) + kron(eye(size(G.W, 1)), time_W);
A_max = max(max(A));
A = A/A_max;
%%
m = [0.5:0.1:0.9,0.995];  %Sampling density
SampleMatrix = squeeze(sample_matrices(4,:,:));
J = reshape(SampleMatrix, [size(SampleMatrix, 1)*size(SampleMatrix, 2), 1]);
indx_non_sampled = find(SampleMatrix(:) == 0);
x_vector_original = x_matrix(indx_non_sampled);
x_vector_sampled = x_matrix.*SampleMatrix;

%%
num_taps = 2;
x_vector_sampled_flatten = reshape(x_vector_sampled, [size(x_vector_sampled, 1)*size(x_vector_sampled, 2), 1]);
X_unrolled = [ones(size(x_vector_sampled_flatten))];
for i = 1:num_taps+1
    X_unrolled = [X_unrolled, A^(i-1)*x_vector_sampled_flatten];
end
for k1 = 1:num_taps+1
    for k2 = 1:num_taps+1
        X_unrolled = [X_unrolled, A^(k1-1)*x_vector_sampled_flatten.*A^(k2-1)*x_vector_sampled_flatten];
    end
end
X_unrolled_max = max(max(X_unrolled));
% X_unrolled = X_unrolled/X_unrolled_max;
%%
H = abs(rand(size(X_unrolled, 2),1))/100;
error = zeros(size(x_vector_sampled_flatten));

%%
% hyperparameters
% mu = 3e-5;
epochs = 1e6;
mu_array = [7e-4, 1e-3, 1e-3, 1e-3, 1e-3];
lambda = 0;
format short g;
% gradient descent
for i = 1:epochs
    error = x_vector_sampled_flatten - (X_unrolled*H).*J;
    mu = mu_array(ceil((i/epochs)*length(mu_array)));

    H = H + mu*(X_unrolled/X_unrolled_max)'*error;

%     H(2) = H(2) - lambda*2*abs(H(2))*(-1)^(H(2)<0);
    % H = H - 2*lambda*H;
    
    H(2) = 0;
    
    if (rem(i,1000)==0)
        disp([i, norm(error), norm(x_vector_sampled_flatten-X_unrolled*H), lambda*(H'*H), mu]);
    end
    
end
x_recon = reshape(X_unrolled*H, size(SampleMatrix,1), size(SampleMatrix,2));
x_vector_reconstructed = x_recon(indx_non_sampled);
disp(sqrt(mean((x_vector_original-x_vector_reconstructed).^2))*x_matrix_max)

figure;
stem(reshape(x_matrix, [size(x_matrix, 1)*size(x_matrix, 2), 1]));
hold on;
stem(X_unrolled*H);
hold on;
stem(x_vector_sampled_flatten);
legend('original', 'reconstructed', 'sampled');










