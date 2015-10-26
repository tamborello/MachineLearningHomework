function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.1, 0.3, 1, 3, 10, 30, 100, 300];
sigma_set = C_set * .1;
temp = zeros(length(C_set)^2, 3);
idx = 0;
for c = C_set;
	for s = sigma_set;
	idx = idx + 1
% model
		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
% prediction
		prediction = svmPredict(model, Xval);
% error
		temp(idx, 1) = mean(double(prediction ~= yval));
% that c
		temp(idx, 2) = c;
% that sigma
		temp(idx, 3) = s;
	endfor
endfor

[x, ix] = min(temp(:,1));

C = temp(ix,2);
sigma = temp(ix,3);

% =========================================================================

end
