function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% Flag to check if z is a vector
VECTORFLAG = 0;

if size(z,2)==1,

    VECTORFLAG = 1;
    z = z';

end

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:size(z,2),

    g(i) = 1 / (1 + exp(-1 * z(i)));

end

if VECTORFLAG==1,

    g = g';

end

% =============================================================

end
