function [error_test] = computeTest(Xtest, ytest, lambda_test)
% After training and validating a model's theta and lambda parameters, test them.

% We used 8th degree polynomial features in training & validation, so use 8 here
Xtest_poly = polyFeatures(Xtest, 8);

theta_test = trainLinearReg([ones(size(Xtest_poly,1), 1) Xtest_poly], ytest, lambda_test);
error_test = linearRegCostFunction([ones(size(Xtest_poly,1), 1) Xtest_poly], ytest, theta_test, 0);

end