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

temp1 = zeros(size(Theta1, 1), size(Theta1, 2) - 1);
temp2 = zeros(size(Theta2, 1), size(Theta2, 2) - 1);

for (j = 1:size(Theta1, 1));
	for (k = 2:size(Theta1, 2));
		temp1(j,k) = Theta1(j,k)^2;
	endfor;
endfor;
		
for (j = 1:size(Theta2, 1));
	for (k = 2:size(Theta2, 2));
		temp2(j,k) = Theta2(j,k)^2;
	endfor;
endfor;

J = (m ^ -1) * sum(sum(J)) + ...
	lambda / (2 * m) * (sum(sum(temp1)) + sum(sum(temp2)));
	


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);


% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 

% grad = sigmoidGradient([Theta1(:) ; Theta2(:)]);
% I think I'll need to reshape grad into Theta1- & Theta2-sized matrices before I use it
% in backpropagation

% a1 = zeros(size(X, 2) + 1, 1);
% a2 = zeros(size(Theta1, 1) + 1, 1);
% a3 = zeros(size(Theta2, 1), 1);
% delta3 = a3;
% delta2 = a2;
% delta1 = a1;
Delta1 = zeros(hidden_layer_size, (1 + input_layer_size));
Delta2 = zeros(num_labels, (1 + hidden_layer_size));

for (t = 1:m);
% feedforward
	a1 = [1; X(t, :)'];
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
% feedbackward
delta3 = a3 - newY(t,:)';
delta2 = Theta2'(2:end,:) * delta3 .* sigmoidGradient(z2);
% Did I remove delta-2-0 at the correct time?
Delta2 = Delta2 + delta3 * a2';
Delta1 = Delta1 + delta2 * a1';

endfor;

% unregularized gradient
% grad = (1 / m) * [Delta1(:); Delta2(:)];

% regularized gradient
grad = ... % for each l, j = 0, then j > 0
	[(1 / m) * Delta1(:,1); ...
	(1 / m) * Delta1(:,2:end)(:) + (lambda / m) * Theta1(:, 2:end)(:); ...
	(1 / m) * Delta2(:,1); ...
	(1 / m) * Delta2(:,2:end)(:) + (lambda / m) * Theta2(:, 2:end)(:)];
	
end
