function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

#nn 1
b1_mat = repmat(stack{1}.b, 1, M);
z2 = stack{1}.w * data + b1_mat;
a2 = 1 ./ (1 + exp(-z2));

#nn 2
b2_mat = repmat(stack{2}.b, 1, M);
z3 = stack{2}.w * a2 + b2_mat;
a3 = 1 ./ (1 + exp(-z3));

#softmax layer
soft_z = softmaxTheta * a3;
soft_z = bsxfun(@minus, soft_z, max(soft_z, 1));
soft_exp = exp(soft_z);
soft_pred = bsxfun(@rdivide, soft_exp, sum(soft_exp, 1));

cost = -sum(log(soft_pred(find(groundTruth == 1)))); 

%for i=1:M
%  cost += -log(soft_pred(labels(i), i)); 
%end
cost /= M;

cost += lambda /2.0 * sum(sum(softmaxTheta .* softmaxTheta, 1), 2);

%softmax thetagrad
softmaxThetaGrad = -1/M * (groundTruth - soft_pred) * a3' + lambda * softmaxTheta;

fprime_z3 = a3 .* (1 - a3);  %
fprime_z2 = a2 .* (1 - a2);

up_soft = -1.0 * softmaxTheta' * (groundTruth - soft_pred); % theta' * (I - P)
delta3 = up_soft .* fprime_z3;
delta2 = stack{2}.w' * delta3 .* fprime_z2;

W2grad = delta3 * a2' / M;
W1grad = delta2 * data' / M;

b2grad = sum(delta3, 2) / M;
b1grad = sum(delta2, 2) / M;

stackgrad{2}.w = W2grad;
stackgrad{2}.b = b2grad;

stackgrad{1}.w = W1grad;
stackgrad{1}.b = b1grad;
% 


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%















% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
