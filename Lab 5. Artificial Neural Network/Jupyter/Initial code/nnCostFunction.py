import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value):
#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)
#   computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
    tmp = nn_params.copy()
    Theta1 = np.reshape(tmp[0:hidden_layer_size * (input_layer_size + 1)], 
                          (hidden_layer_size, (input_layer_size + 1)), order='F')
    Theta2 = np.reshape(tmp[(hidden_layer_size * (input_layer_size + 1)):len(tmp)], 
                          (num_labels, (hidden_layer_size + 1)), order='F')

# Setup some useful variables
    m = np.shape(X)[0]

# Computation of the Cost function including regularisation
# Feedforward 
    a2 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), X)), np.transpose(Theta1)))
    a3 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), a2)), np.transpose(Theta2)))

    # Cost function for Logistic Regression summed over all output nodes
    Cost = np.empty((num_labels, 1))
    for k in range(num_labels):
        # which examples fit this label
        y_binary=(y==k+1)
        # select all predictions for label k
        hk=a3[:,k]
        # compute two parts of cost function for all examples for node k
        Cost[k][0] = np.sum(np.transpose(y_binary)*np.log(hk)) + np.sum(((1-np.transpose(y_binary))*np.log(1-hk)))
        
# Sum over all labels and average over examples
    J_no_regularisation = -1./m * sum(Cost)
# No regularization over intercept
    Theta1_no_intercept = Theta1[:, 1:]
    Theta2_no_intercept = Theta2[:, 1:]

# Sum all parameters squared
    RegSum1 = np.sum(np.sum(np.power(Theta1_no_intercept, 2)))
    RegSum2 = np.sum(np.sum(np.power(Theta2_no_intercept, 2)))
# Add regularisation term to final cost
    J = J_no_regularisation + (lambda_value/(2*m)) * (RegSum1+RegSum2)

# You need to return the following variables correctly 
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

# ====================== YOUR CODE HERE ======================
# Implement the backpropagation algorithm to compute the gradients
# Theta1_grad and Theta2_grad. You should return the partial derivatives of
# the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# Theta2_grad, respectively. After implementing Part 2, you can check
# that your implementation is correct by running checkNNGradients
#
# Note: The vector y passed into the function is a vector of labels
#       containing values from 1..K. You need to map this vector into a 
#       binary vector of 1's and 0's to be used with the neural network
#       cost function.
#
# Hint: It is recommended implementing backpropagation using a for-loop
#       over the training examples if you are implementing it for the 
#       first time.
#

    D_Th1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    D_Th2 = np.zeros((num_labels, hidden_layer_size + 1))
    
    for t in range(0,len(X)):

            # step 1
            a_1 = X[t]
            a_1 = np.append([1], a_1)
            a_2 = np.insert(a2[t], 0, 1)
            a_3 = a3[t]

            #step 2
            yk = np.zeros(num_labels)
            yk[y[t]-1] = 1

            delta_3 = a_3 - yk

            #step 3
            delta_2 = (np.matmul(Theta2.transpose(), delta_3) * a_2 * (1 - a_2))[1:]

            # step 4
            D_Th1 = D_Th1 + np.outer(delta_2, a_1.transpose())
            D_Th2 = D_Th2 + np.outer(delta_3, a_2.transpose())
            
   # step 5
    for Ds, Thetas in [D_Th1, Theta1], [D_Th2, Theta2]:
        Ds/=m
        Ds[:,1:] += Thetas[:,1:] * lambda_value / m
        
        
    Theta1_grad = D_Th1
    Theta2_grad = D_Th2



# -------------------------------------------------------------

# =========================================================================

# Unroll gradients
    Theta1_grad = np.reshape(Theta1_grad, Theta1_grad.size, order='F')
    Theta2_grad = np.reshape(Theta2_grad, Theta2_grad.size, order='F')
    grad = np.expand_dims(np.hstack((Theta1_grad, Theta2_grad)), axis=1)
    
    return J, grad
