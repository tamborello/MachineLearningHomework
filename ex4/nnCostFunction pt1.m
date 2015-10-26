function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Re-encode y so that it's neural network compatible.
newY = zeros(m, num_labels);
for (i = 1:m); newY(i, y(i)) = 1; endfor;
% J = newY;
% Man, that took forever! How could I do this using a vectorized method?
% There are some hints about using an identity matrix.

J = zeros(m, num_labels);

for (k = 1:num_labels);
	J (:,k) = log(feedforward(X, Theta1, Theta2)(:,k)) .* -newY(:,k) ...
	- log(1 - feedforward(X, Theta1, Theta2)(:,k)) .* (1 - newY(:,k));
 endfor;

J = (m ^ -1) * sum(sum(J));




% hypothesis = sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2');

% temp1 = zeros(num_labels, 1);
% temp2 = zeros(m, 1);

% for each example
% for (i = 1:m);
% cost per output node
%	for (j = 1:num_labels);
%		temp1(j) = ...
%		-newY(i,j) * log(sigmoid([ones(m, 1) sigmoid(X(i,:) * Theta1')] * Theta2')) ...
%		- (1 - newY(i,j)) * log(1 - sigmoid([ones(m, 1) sigmoid(X(i,:) * Theta1')] * Theta2'));
%	endfor;
% sum across the output nodes
% J =	size(sum(temp1))
%	temp2(i) = sum(temp1(:));
% endfor;
% sum across the examples
% size(sum(temp2))
% J = sum(temp2);
% Jeez this is ugly & slow! How might I vectorize it?

% What if I just iterate through the examples?
% temp = zeros(m, 1);
% for (i = 1:m);
%	temp(i) = -newY(i) * log(sigmoid([ones(m, 1) sigmoid(X(i,:) * Theta1')] * Theta2')) ...
%	- (1 - newY(i) * log(sigmoid([ones(m, 1) sigmoid(X(i,:) * Theta1')] * Theta2')))
% endfor;
% J = sum(temp);

% log(sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2'))

% J = (m ^ -1) * ...
%	((-newY' * log(sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2'))) - ...
%	((1 - newY') * ...
%	log(1 - sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2'))));
% sum the cost over all m


% J = sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2');













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
