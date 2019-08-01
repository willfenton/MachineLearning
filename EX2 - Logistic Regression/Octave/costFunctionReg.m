function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
p1 = (-1) * y' * log(h);
p2 = (-1) * (1 - y)' * log(1 - h);

p3 = theta;
p3(1) = 0;
reg_term = (lambda / (2 * m)) * sum(p3.^2);

J = (1 / m) * (p1 + p2) + reg_term;

reg_vec = theta * (lambda / m);
reg_vec(1) = 0;

grad = ((1 / m) * X' * (h - y)) + reg_vec;

% =============================================================

end
