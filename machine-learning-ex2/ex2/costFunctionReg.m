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

sigmoid1 = sigmoid(theta'*X');

J = (log(sigmoid1) * (-y) - log(sigmoid1 -1) *(1 - y)) / m + lambda / (2 * m)*sum(theta(2:end).^2);

grad = (sigmoid1 - y') * X / m + ( lambda / m ) * theta' ;
grad(1) = ((sigmoid1 - y') * X(:,1)) / m ;

% =============================================================

end
