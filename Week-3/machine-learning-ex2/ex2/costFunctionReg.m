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

n = size(theta,1);

prediction = X*theta;
prediction = sigmoid(prediction);

for i=1:m,
	J = J + y(i)*log(prediction(i)) + (1-y(i))*log(1-prediction(i));
end

J = (-1/m)*J;

regularizationfactor = 0;

for k=2:n,
    regularizationfactor = regularizationfactor + theta(n)^2;
end

regularizationfactor = (lambda/(2*m)) * regularizationfactor;

J = J + regularizationfactor;

for i = 1:size(theta,1),
    for j = 1:m,

        grad(i) = grad(i) + (prediction(j)-y(j))*X(j,i);

    end
end

grad = (1/m).*grad;

for k=2:n,
    grad(k) = grad(k) + (lambda/m)*theta(k);
end



% =============================================================

end
