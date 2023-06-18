function y = volterra_filter(x, A, L, C)
%VOLTERRA_FILTER Summary of this function goes here
%   Detailed explanation goes here
c0 = C(1);
c1 = C(2:L+1);
c2 = reshape(C(L+2:end), [L, L]);

y = c0 * ones(size(x));

for k = 1:L
    y = y + c1(k)*(A^(k-1))*x;
end

for k1 = 1:L
    for k2 = 1:L
        y = y + c2(k1, k2)*((A^(k1-1))*x) .* ((A^(k2-1))*x);
    end
end

end