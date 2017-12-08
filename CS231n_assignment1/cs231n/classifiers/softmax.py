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
  num_class = W.shape[1]
  num_dim = W.shape[0]
  
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    e = np.exp(scores)
    norm_scores = e / np.sum(e)
    loss += -np.log(norm_scores[y[i]])
    for j in range(num_class):
      if j == y[i]:
        dW[:, j] = dW[:, j] + (e[y[i]] / np.sum(e) - 1) * X[i]
      else:
        dW[:, j] = dW[:, j] + e[j] / np.sum(e) * X[i]

  loss /= num_train
  loss += reg * np.sum(W ** 2)

  dW /= num_train
  dW += reg * W
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
  num_class = W.shape[1]
  num_dim = W.shape[0]

  XW = X @ W
  E = np.exp(XW)
  norm_E = E / np.sum(E, axis=1).reshape((num_train, 1))
  idx = num_class * np.arange(num_train) + y
  loss += np.sum(-np.log(norm_E).flat[idx])
  loss /= num_train
  loss += reg * np.sum(W ** 2)

  norm_E.flat[idx] -= 1
  dW = np.transpose(X) @ norm_E / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

