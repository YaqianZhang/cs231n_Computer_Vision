import numpy as np
from random import shuffle

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
  num_data=X.shape[0]
  dim=X.shape[1]
  num_classes=W.shape[1]
  scores= np.dot(X,W)
  scores= (scores.T -np.max(scores,axis=1)).T
  for i in range(num_data):  
        loss+= -scores[i,y[i]]
        p=0.0
        dw_oneSample=np.zeros([dim,num_classes])
        for j in range(num_classes):
            f=np.exp(scores[i,j])
            p+=f
            dw_oneSample[:,j] = f*X[i,:].T
        loss += np.log(p)
        dw_oneSample /= p
        dW += dw_oneSample
        dW[:,y[i]] += -X[i,:].T

  loss /= num_data
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_data
  dW += reg*W

 

                 
        
    
    

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
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]
  scores= X.dot(W)#N*C
  scores= (scores.T -np.max(scores,axis=1)).T
    
  s1=np.exp(scores)#N*C
  
  #s2=np.sum(s1,axis=1)#N
  s2=s1.dot(np.ones([C,1]))
  s3=np.log(s2)#N
  s4=np.sum(s3)# 1
  s5=scores[range(N),y]#N
  #s6=np.sum(s5) #1
  s6 =np.ones([1,N]).dot(s5)
  loss=-s6+s4
  loss /= N 
  loss += 0.5*reg*np.sum(W*W)
   
  delta_s4=1
  delta_s6=-1
  delta_s5=delta_s6 *np.ones([N,1])
  delta_s3=delta_s4*np.ones([N,1])
  delta_s2=delta_s3*(1.0/s2) # N
  delta_s1=delta_s2.dot(np.ones([1,C])) # N*C
  delta_scores=delta_s1 * s1# N*C
  delta_W=X.T.dot(delta_scores)# D*C
  
  #delta_W=X.T.dot(s1/s2)
  Y=np.zeros([N,C])
  Y[range(N),y]=1
  delta_W +=-X.T.dot(Y)
  dW=delta_W
  dW /= N
  dW += reg*W


    
  
  
   

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

