from numpy import *
#import scipy.io as sio
from backwards import backwards
from costFunction import costFunction

def fmincgb(theta, X, y, layers, num_labels, lamb, iter, tol):
    # cgbt2: Conjugate gradient descent method with backtracking line search
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # input_layer_size / hidden_layer_size / num_labels: As defined in neural network
        # lamb: Regularization variable
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iter: Maximum number of iterations
        # tol: The procedure will break if the square of the Newton decrement is less than the threshold tol
    alpha = 0.25
    beta = 0.5

    # Initialize the gradient
    dxPrev = -backwards(theta, layers, X, y,num_labels, lamb)
    snPrev = dxPrev
    #  theta = np.matrix(theta).T
    # Iteration
    for i in range(iter):   
        J    = costFunction(theta, layers, X, y, num_labels, lamb)
        grad = backwards(theta, layers, X, y,num_labels, lamb)

        dx = -grad
        if dot(dx.transpose(),dx) < tol:
            print('Terminated due to stopping condition with iteration number', i)
            return theta
        # betaPR since beta is already used as backtracking variable
        # Polak-Ribiere formula
        betaPR = max((0,(dot(dx.transpose(),(dx-dxPrev)))/(dot(dxPrev.transpose(),dxPrev))))
        # Search direction
        sn = dx + snPrev*betaPR
        # Backtracking
        t = 1
        costNew = costFunction((theta + t*sn), layers, X, y, num_labels, lamb)
        alphaGradSn = alpha*dot(grad.transpose(),sn)
        while costNew > J+t*alphaGradSn or isnan(costNew):
            t = beta*t
            
            costNew = costFunction(theta+t*sn,layers, X, y, num_labels, lamb)

        tRight = t*2
        tTemp = t
        while tRight-t > 1e-3: # Search right-hand side
            costRight = costFunction(theta+tRight*sn,layers, X, y, num_labels, lamb)
            if costRight > costNew:
                tRight = (t+tRight)/2
            else:
                t = tRight
                tRight = 2*t
                costNew = costRight

        if t == tTemp:
            tLeft = t/2.0
            while t-tLeft > 1e-3: # Search left-hand side
                costLeft = costFunction(theta+tLeft*sn,layers, X, y, num_labels, lamb)
                if costLeft > costNew:
                    tLeft = (t+tLeft)/2
                else:
                    t = tLeft
                    tLeft = t/2
                    costNew = costLeft

        # Update
        theta += t*sn
        snPrev = sn
        dxPrev = dx
        print('Iteration',i+1,' | Cost:',costNew)

    return theta
