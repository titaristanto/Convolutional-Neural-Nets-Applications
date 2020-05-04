import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
  # Right now the loss and gradient are the sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  delta = 1

  scores = np.dot(X, W)

  # Compute scores of correct label for all data points
  correct_label_scores = scores[range(num_train), y]

  # Remove unnecessary surplus score at j=y_i. The score s_y and s_j will cancel out later, leaving only delta
  scores[range(num_train), y] -= delta

  # Compute loss (including regularization term).
  difference = scores.T - correct_label_scores + delta
  sum_loss = np.maximum(np.zeros((num_classes, num_train)), difference)
  loss = np.sum(sum_loss)/num_train + reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad = difference

  # Since negative loss is thresholded to 0, and thus the derivative is 0, we can formulate:
  grad[grad <= 0] = 0
  grad[y, range(num_train)] = 0 # zero out the correct label

  # The derivative of L or (s_j - s_y + delta) with respect to s_j is 1
  grad[grad > 0] = 1

  # The derivative of L or (s_j - s_y + delta) w.r.t. s_y is -1. Since it is repeated
  # j times in every data point, it is basically equal to the negative sum of grad in every data point
  temp = -np.sum(difference, axis=0)
  grad[y, range(num_train)] = temp

  # Compute gradient matrix (including regularization term)
  dW = np.transpose(np.dot(grad, X))/num_train + 0.5*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
