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

sigmat = sigmoid(X * theta);
difmat = y .* log(sigmat) + (1-y) .* log(1-sigmat);
J = -1*sum(difmat)/m;
regmat = zeros(m,1);
regmat = (theta.^2);
regmat(1,1)=0;
J += sum(regmat)*lambda/(2*m);

grad = X' * (sigmat - y)/m;
regdif = theta;
regdif(1,1) = 0;
grad += regdif*lambda/m;




% =============================================================

end
