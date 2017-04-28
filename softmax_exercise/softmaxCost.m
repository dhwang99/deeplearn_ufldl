function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
thetagrad = zeros(numClasses, inputSize);

cost = 0;
J = 0;

M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
predictM = bsxfun(@rdivide, M, sum(M, 1)); 

for i=1:numCases
  J += -log(predictM(labels(i), i));
end

J = J / numCases;
J += sum(sum(theta .* theta, 1), 2) * lambda / 2;
cost = J;

partJ_Z = predictM - groundTruth;

thetagrad = partJ_Z * data';  % sum(xi*(yi - p(yi|xi;theta)))/m

thetagrad /= numCases;
thetagrad = thetagrad + lambda * theta;

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.












% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

