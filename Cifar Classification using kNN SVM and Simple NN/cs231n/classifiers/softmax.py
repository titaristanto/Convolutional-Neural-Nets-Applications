import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    lin_score = np.dot(W.T, X[i].T) # Cx1

    # Subtract by max exp_score; this is a trick to avoid overflow
    exp_score = np.exp(lin_score - np.max(lin_score))
    total_exp_score = np.sum(exp_score)

    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += -X[i, :] + X[i, :]*exp_score[j]/total_exp_score
      else:
        dW[:, j] += X[i, :]*exp_score[j]/total_exp_score
    loss += -np.log(exp_score[y[i]] / total_exp_score)

  loss = loss/num_train + reg*np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  lin_score = np.dot(X, W).T # CxN
  exp_score = np.exp(lin_score - np.max(lin_score, axis=0, keepdims=True))
  total_exp_score = np.sum(exp_score, axis=0, keepdims=True)
  norm_exp_score = exp_score/total_exp_score

  # computing loss
  loss = np.sum(-np.log(norm_exp_score[y, range(num_train)]))/num_train + reg*np.sum(W*W)

  # computing gradient
  norm_exp_score[y, range(num_train)] -= 1
  dW = np.dot(X.T, norm_exp_score.T)/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

