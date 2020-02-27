import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,use_batchnorm= False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size,filter_size)
    self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W/4,hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)

    
    
    if self.use_batchnorm:
      
      self.params['gamma1'] = np.ones([1,num_filters])
      self.params['beta1'] = np.zeros([1,num_filters])
      self.params['gamma2'] = np.ones([1,hidden_dim])
      self.params['beta2'] = np.zeros([1,hidden_dim])
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]
      #self.bn_params[1] = {'mode': 'train'}
      #self.bn_params[2] = {'mode': 'train'}
      
                                    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

        
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    use_batchnorm = self.use_batchnorm

    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
        gamma1 = self.params['gamma1']
        beta1 = self.params['beta1']
        gamma2 = self.params['gamma2']
        beta2 = self.params['beta2']
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
        

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    a, conv_cache = conv_relu_pool_forward(X,W1,b1,conv_param, pool_param)
    if(use_batchnorm):
        a_bn,bn_cache1 = spatial_batchnorm_forward(a,gamma1,beta1,self.bn_params[0])
        a=np.copy(a_bn)
    N= a.shape[0]
    a1, affine_relu_cache = affine_relu_forward(np.reshape(a,(N,-1)), W2, b2)
    if(use_batchnorm):
        a1_bn,bn_cache2 = batchnorm_forward(a1,gamma2,beta2,self.bn_params[1])
        a1=np.copy(a1_bn)
    scores, affine_cache = affine_forward(a1, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, dscores = softmax_loss(scores,y)
    loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    
    dx3, dW3, db3 = affine_backward(dscores,affine_cache)
    if(use_batchnorm):
        dx3_bn,dgamma2, dbeta2 = batchnorm_backward(dx3,bn_cache2)
        dx3 = np.copy(dx3_bn)
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2
    
    dx2, dW2, db2 = affine_relu_backward(dx3, affine_relu_cache)

    [N,C,H,W]=X.shape
    F=W1.shape[0]
    dx1 = np.reshape(dx2,(N,F,H/2,W/2))

    
    if(use_batchnorm):
        dx1_bn,dgamma1,dbeta1 = spatial_batchnorm_backward(dx1,bn_cache1)
        dx1 = np.copy(dx1_bn)
        grads['gamma1']=dgamma1
        grads['beta1'] = dbeta1

    dx, dW1, db1 = conv_relu_pool_backward(dx1, conv_cache)
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    
    
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

class FullConvNet(object):
    
   def __init__(self, input_dim=(3, 32, 32), num_filters = [32], filter_size =[3], hidden_dims = [100], num_classes=10, 
               weight_scale=1e-3, reg=0.0,use_batchnorm= False, use_regression = False,
               dtype=np.float32):
    #con_relu_outside=False, con_size_one_unit=1, 
    """
    Initialize a new network.
    [conv-relu-pool]XN - [affine-relu]XM - affine -[softmax or SVM]
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: A list of integers giving the number of filters to use in each convolutional layer (N)
    - filter_size: A list of integers giving the size of filters to use in each convolutional layer (N)
    - hidden_dims: A list of integers giving the size of each hidden layer (M)
    - num_classes: Number of scores to produce from the final affine layer.
        !!In regression this is just the last dim of last layer
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_regression = use_regression
    C, H, W = input_dim
    num_con = len(num_filters)
    num_affine = len(hidden_dims)+1
    self.num_con = num_con
    self.num_affine = num_affine
    self.filter_size = filter_size
    
    
    
    channels=C
    for i in xrange(num_con):
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(num_filters[i], channels, filter_size[i],filter_size[i])
        self.params['b'+str(i+1)] = np.zeros(num_filters[i])
        channels = num_filters[i]
        H /= 2
        W /= 2
        if self.use_batchnorm:
            self.params['gamma'+str(i+1)] = np.ones([1,channels])
            self.params['beta'+str(i+1)] = np.zeros([1,channels])
    out_dim_from_con = channels * H * W
    

    

    affine_dims = hidden_dims
    affine_dims.append(num_classes)
    prev_dim = out_dim_from_con
    start = num_con + 1
    for i in xrange(num_affine):
        self.params['W' + str(start+i)] = weight_scale * np.random.randn(prev_dim,affine_dims[i])
        self.params['b' + str(start+i)] = np.zeros(affine_dims[i])
        prev_dim = affine_dims[i]
        if self.use_batchnorm:
            if(i<(num_affine-1)):
                self.params['gamma'+str(start+i)] = np.ones([1,prev_dim])
                self.params['beta'+str(start+i)] = np.zeros([1,prev_dim])
        
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(num_con + num_affine -1)]

      

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    
   def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    use_batchnorm = self.use_batchnorm    
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    #######################################    Forward Pass    ##############################
    scores = None    
    num_con = self.num_con
    num_affine = self.num_affine
    filter_size = self.filter_size
    start = self.num_con +1 # also used in backward pass
    cache = []
    I = np.copy(X)
    for i in xrange(num_con):
        W, b = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
        conv_param = {'stride': 1, 'pad': (filter_size[i] - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        a, conv_cache = conv_relu_pool_forward(I,W,b,conv_param, pool_param)
        cache.append(conv_cache)
        if(use_batchnorm):
            gamma, beta = self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)]
            a_bn,bn_cache1 = spatial_batchnorm_forward(a,gamma,beta,self.bn_params[i])
            a=np.copy(a_bn)
            cache.append(bn_cache1)
        I = np.copy(a)
    Nc,Cc,Hc,Wc= I.shape # also used in backward pass
    
    I2 = np.reshape(I,(Nc,-1)) 
    prev = np.copy(I2)
    
    for i in xrange(num_affine):
        W, b = self.params['W'+str(start+i)], self.params['b'+str(start+i)]
        if(i == (num_affine-1)):
            a1, affine_cache = affine_forward(prev, W, b)
            cache.append(affine_cache)
        else:
            a1, affine_relu_cache = affine_relu_forward(prev, W, b)
            cache.append(affine_relu_cache)
            if(use_batchnorm):
                gamma, beta = self.params['gamma'+str(start +i)], self.params['beta'+str(start + i)]
                a1_bn,bn_cache2 = batchnorm_forward(a1,gamma,beta,self.bn_params[start+i-1])
                cache.append(bn_cache2)   
                a1=np.copy(a1_bn)
        prev = np.copy(a1)
    scores = np.copy(prev)
    
    if(~self.use_regression):
        if y is None:
          return scores
    #######################################    Backward Pass    ############################## 
    loss, grads = 0, {}
    reg = self.reg
    if(self.use_regression):
        loss, dscores = rmse_loss(scores,y)
    else:
        loss, dscores = softmax_loss(scores,y)
        
    if(self.use_regression):
        if y is None:
          return loss
       
        
    
    
    
    dout = np.copy(dscores)

    
    for i in xrange(num_con + num_affine):
        w = self.params['W'+str(i+1)]
        loss += 0.5 * reg * np.sum(w*w)
        
    for i in reversed(xrange(num_affine)):
        
        if(i == (num_affine-1)):
            dx, dw, db = affine_backward(dout,cache.pop())
        else:
            if(use_batchnorm):
                dx, dgamma, dbeta = batchnorm_backward(dout,cache.pop())
                dout = np.copy(dx)
                grads['gamma'+str(start +i)] = dgamma
                grads['beta' + str(start +i)] = dbeta
            dx, dw, db = affine_relu_backward(dout,cache.pop())   
        dout = np.copy(dx)
        grads['W'+str(start+i)] = dw + reg * self.params['W' + str(start+i)]
        grads['b'+str(start+i)] = db
    dout_a = np.reshape(dout, I.shape)
    dout_c = np.copy(dout_a)
    for i in reversed(xrange(num_con)):
        if(use_batchnorm):
            dx, dgamma, dbeta = spatial_batchnorm_backward(dout_c,cache.pop())
            grads['gamma'+str(i+1)] = dgamma
            grads['beta' + str(i+1)] = dbeta
            dout_c = np.copy(dx)
        dx, dw, db = conv_relu_pool_backward(dout_c,cache.pop())
        dout_c = np.copy(dx)
        grads['W'+str(i+1)] = dw + reg * self.params['W' + str(i+1)]
        grads['b'+ str(i+1)] = db 
         
        
            
    
    
    return loss,grads
            
            
           

        

    '''    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    if self.use_batchnorm:
        gamma1 = self.params['gamma1']
        beta1 = self.params['beta1']
        gamma2 = self.params['gamma2']
        beta2 = self.params['beta2']
    
    

    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    '''

    

        

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    '''
    a, conv_cache = conv_relu_pool_forward(X,W1,b1,conv_param, pool_param)
    if(use_batchnorm):
        a_bn,bn_cache1 = spatial_batchnorm_forward(a,gamma1,beta1,self.bn_params[0])
        a=np.copy(a_bn)
    N= a.shape[0]
    a1, affine_relu_cache = affine_relu_forward(np.reshape(a,(N,-1)), W2, b2)
    if(use_batchnorm):
        a1_bn,bn_cache2 = batchnorm_forward(a1,gamma2,beta2,self.bn_params[1])
        a1=np.copy(a1_bn)
    scores, affine_cache = affine_forward(a1, W3, b3)
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, dscores = softmax_loss(scores,y)
    loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    
    dx3, dW3, db3 = affine_backward(dscores,affine_cache)
    if(use_batchnorm):
        dx3_bn,dgamma2, dbeta2 = batchnorm_backward(dx3,bn_cache2)
        dx3 = np.copy(dx3_bn)
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2
    
    dx2, dW2, db2 = affine_relu_backward(dx3, affine_relu_cache)

    [N,C,H,W]=X.shape
    F=W1.shape[0]
    dx1 = np.reshape(dx2,(N,F,H/2,W/2))

    
    if(use_batchnorm):
        dx1_bn,dgamma1,dbeta1 = spatial_batchnorm_backward(dx1,bn_cache1)
        dx1 = np.copy(dx1_bn)
        grads['gamma1']=dgamma1
        grads['beta1'] = dbeta1

    dx, dW1, db1 = conv_relu_pool_backward(dx1, conv_cache)
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    return loss, grads
    '''
    
    
