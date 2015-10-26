% initialize theta
function [initial_theta] = initializeTheta(X)

[m, n] = size(X);
initial_theta = zeros(n, 1);