function [g, cost] = grad(x_sample, C_pred, A_strong, L, J)
%GRADIENT Summary of this function goes here
%   Detailed explanation goes here
dx = 0.001;
g = zeros(size(C_pred));
cost = sum((volterra_filter(x_sample, A_strong, L, C_pred).*J - x_sample).^2);

for k = 1:length(C_pred)
    dH_0 = zeros(size(C_pred));
    dH_0(k) = dx;
    C_plus = C_pred + dH_0;
    C_minus = C_pred - dH_0;
    g(k) = (sum((volterra_filter(x_sample, A_strong,L, C_plus).*J - x_sample).^2) - sum((volterra_filter(x_sample,A_strong,L, C_minus).*J - x_sample).^2))/(2*dx);
end
