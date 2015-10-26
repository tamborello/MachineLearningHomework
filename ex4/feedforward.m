function h = feedforward(X, Theta1, Theta2)
% function feedforward computes the hypothesis of a feedforward neural network having one
% hidden layer
m = size(X, 1);

h = sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2');

end