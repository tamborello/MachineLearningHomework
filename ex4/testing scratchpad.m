[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% small test network
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';
nn_params = [Theta1(:) ; Theta2(:)];


[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda)

size(-y(1,1) * log(sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2')) - (1 - y(1,1)) * log(1 - sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2')))
