import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    
# You need to return the following variables correctly 
    p = np.zeros(m);

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#


    X0 = np.ones((m,1))
    a_1 = np.hstack((X0,X))

    z_2 = np.matmul(a_1,Theta1.transpose())
    a_2 = sigmoid(z_2)
    
    a_2 = np.hstack((X0,a_2))

    z_3 = np.matmul(a_2,Theta2.transpose())
    a_3 = sigmoid(z_3)
    
    p = a_3
    p = np.argmax(p, axis=1)


    return p

# =========================================================================
