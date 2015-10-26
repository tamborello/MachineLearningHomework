function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% J = sum((((theta(1) * X(:,1)) + (theta(2:end) * X(:,2:end))) - y) .^ 2) / (2 * m) ...
%	+ lambda/(2*m) * sum(theta(2:end) .^ 2);

% Multiply X & theta in the order X, then theta, as per the HW1 pitfalls document
%								10x2			2x1
J = sum((((X(:,1) * theta(1)) + (X(:,2:end) * theta(2:end))) - y) .^ 2) / (2 * m) ...
	+ lambda/(2*m) * sum(theta(2:end) .^ 2);



% =========================================================================
% seems to be correct except for dimensions should apparently be 2 x 1
%grad =  sum((((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y) * X(:,1)') / m + ...
%		sum((((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y) * X(:,2:end)') / m + ...
%		lambda/m * [0; theta(2:end)];

% seems to be correct except it doesn't generalize to X and theta of other dimensions
%grad =  [sum((((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y)' * X(:,1)) / m ; ...
%		sum((((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y)' * X(:,2:end)) / m] + ...
% 		lambda/m * [0; theta(2:end)];

% intermittent nonconformant operand error for +
%grad =  [((((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y)' * X(:,1)) / m ; ...
%		(X(:,2:end)' * (((theta(1) * X(:,1)) + (X(:,2:end) * theta(2:end))) - y)) / m] + ...
% 		lambda/m * [0; theta(2:end)];

%			1			21x1	  1 	 		 21x1	->21x1 21x1->1x21  21x2 -> 2x1
% grad =  (((((theta(1) * X(:,1)) + (theta(2:end) * X(:,2:end))) - y)' * X)' / m) + ...
% 		((lambda/m) * [0; theta(2:end)]);
 		
% Multiply X & theta in the order X, then theta, as per the HW1 pitfalls document
%									10x2		2x1
grad =  (((((X(:,1) * theta(1)) + (X(:,2:end) * theta(2:end))) - y)' * X)' / m) + ...
 		((lambda/m) * [0; theta(2:end)]);



grad = grad(:);

end
