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


J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% 
X = [ones(m,1) X]; 
K = num_labels;
for i = 1:m
    yk = zeros(num_labels,1);
    yk(y(i)) = 1;
    a2 = sigmoid(Theta1*X(i,:)');
    a2 = [1; a2];
    a3 = sigmoid(Theta2*a2);
    
    J = J +  sum ( -yk.*log(a3) - (1-yk).*log(1 - a3) ) ;
end
    J = J/m + (lambda/2/m) * (sum ( sum ( Theta1(:,2:end).^2) ) + ...
                             sum ( sum ( Theta2(:,2:end).^2) ));
L = 1;
Delta2 = 0 ;
Delta1 = 0;
for t = 1 : m
    yk = zeros(num_labels,1);
    yk(y(t)) = 1;
    a1 = X(t,:)';
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    
    delta3 = a3 - yk;
    delta2 = Theta2'* delta3.*sigmoidGradient([1; z2]);
    delta2 = delta2(2:end);
    
    Delta2 = Delta2 + delta3*a2';
    Delta1 = Delta1 + delta2*a1';
end
Theta1_grad(:,1) = Delta1(:,1)/m;
Theta2_grad(:,1) = Delta2(:,1)/m;

Theta1_grad(:,2:end) = Delta1(:,2:end)/m + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Delta2(:,2:end)/m + (lambda/m)*Theta2(:,2:end);
    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
    
    
end
