import numpy as np
from random import shuffle

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
#   y=y[:,0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] = dW[:, y[i]] - X[i] # added by jariasf
        dW[:,j] = dW[:,j] + X[i] # added by jariasf

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
#   print(dW)
  dW = dW / num_train # added by jariasf

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW + reg * 2 * W # added by jariasf

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
  # Compute the loss
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
#   correct_class_scores = scores[ np.arange(num_train), y[:,0]].reshape(num_train,1)
  correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
  margin = np.maximum(0, scores - correct_class_scores + 1)
#   margin[ np.arange(num_train), y[:,0]] = 0 # do not consider correct class in loss
  margin[ np.arange(num_train), y] = 0 
  loss = margin.sum() / num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
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
  # Compute gradient
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis=1)
  # Subtract in correct class (-s_y)
#   margin[np.arange(num_train),y[:,0] ] -= valid_margin_count
  margin[np.arange(num_train),y] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train
  
  # Regularization gradient
  dW = dW + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
















#上面jari的方法有誤，問題出在他的y是二維，所以他的correct_class_score就會變成一個很大的矩陣
# 因為一維度跟二維度變成花式索引了
# 可參考我的筆記
# 看y1 y2 y3分別是一維度二維度的矩陣時有啥後果
# 解決的方法也很簡單
# 啊就把y弄成一維度就好
# 就是y[:,0]
