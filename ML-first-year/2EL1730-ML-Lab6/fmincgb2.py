import numpy as np
from backwards import backwards
from costFunction import costFunction

def fmincgb2(X, D, y, layers, num_labels, lamb, iter, tol):

    RHO = 0.01                            # a bunch of constants for line searches
    SIG = 0.5       # RHO and SIG are the constants in the Wolfe-Powell conditions
    INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0                    # extrapolate maximum 3 times the current bracket
    MAX = 20                         # max 20 function evaluations per line search
    RATIO = 100

    length = iter
    red = 1
    
    i = 0                                            # zero the run length counter
    ls_failed = 0                             # no previous line search has failed
    fX = []
    # get function value and gradient
    # [f1 df1] = eval(argstr)
    f1  = costFunction(X, layers, D, y, num_labels, lamb)
    df1 = backwards(X, layers, D, y, num_labels, lamb)
    i = i + (length<0)                                            # count epochs?!
    s = -1.*df1                                       # search direction is steepest
    d1 = -np.dot(s.transpose(),s)                                 # this is the slope
    z1 = red / (1-d1)                                 # initial step is red/(|s|+1)

    while(i < abs(length)):
        if (length>0):
            i = i + 1
        # make a copy of current values
        X0 = X
        f0 = f1
        df0 = df1
        # begin line search
        X = X + z1*s

        f2  = costFunction(X, layers, D, y, num_labels, lamb)
        df2 = backwards(X, layers, D, y,num_labels, lamb)
        if(length < 0):
            i = i + 1 #count epochs?!
        d2 = np.dot(df2.transpose(),s)
        # initialize point 3 equal to point 1
        f3 = f1
        d3 = d1
        z3 = -z1   
    
        if(length > 0):
            M = MAX
        else:
            M = min(MAX, (-length - i))

        # initialize quanteties
        success = False
        limit = -1

        while True:
            while( ((f2 > f1 + z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0)):
                limit = z1
                if(f2 > f1):
                    z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)                   # quadratic fit
                else:
                    A = 6.0*(f2-f3)/z3+3.0*(d2+d3)                                 # cubic fit
                    B = 3.0*(f3-f2)-z3*(d3+2*d2)
                    z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A     #  numerical error possible - ok!

                if( np.isnan(z2) | np.isinf(z2)):
                    z2 = z3/2.0                  # if we had a numerical problem then bisect
                    
                z2 = max(min(z2, INT*z3),(1-INT)*z3)  # don't accept too close to limits
                z1 = z1 + z2                                           # update the step
                X = X + z2*s
                f2  = costFunction(X, layers, D, y, num_labels, lamb)
                df2 = backwards(X, layers, D, y,num_labels, lamb)
                # count epochs?!
                M = M - 1.0
                i = i + (length<0)                          
                d2 = np.dot(df2.transpose(),s)
                z3 = z3-z2     # z3 is now relative to the location of z2
            
            if (f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1):
                break                                                # this is a failure
            elif d2 > SIG*d1:  # success
                success = True
                break
            elif M == 0: # failure
                break

            A = 6.*(f2-f3)/z3 + 3.*(d2+d3)                      # make cubic extrapolation
            B = 3.*(f3-f2) - z3*(d3+2*d2)
            z2 = -d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3))        # num. error possible - ok!

            if (not(np.isreal(z2))) | (np.isnan(z2)) | (np.isinf(z2)) | (z2 < 0):   # num prob or wrong sign?
                if( limit < -0.5):                               # if we have no upper limit
                    z2 = z1 * (EXT-1)                 # the extrapolate the maximum amount
                else:
                    z2 = (limit-z1)/2.0                                  #otherwise bisect
                
            elif (limit > -0.5) & (z2+z1 > limit):          # extraplation beyond max?
                z2 = (limit-z1)/2.0                                               # bisect
            elif (limit < -0.5) & (z2+z1 > z1*EXT): # extrapolation beyond limit
                z2 = z1*(EXT-1.0)                   # set to extrapolation limit
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)):   # too close to limit?
                z2 = (limit-z1)*(1.0-INT)
            # set point 3 equal to point 2
            f3 = f2
            d3 = d2
            z3 = -z2
            # update current estimates
            z1 = z1 + z2
            X = X + z2*s      
            f2  = costFunction(X, layers, D, y, num_labels, lamb)
            df2 = backwards(X, layers, D, y,num_labels, lamb)
            # count epochs?!
            M = M - 1.
            i = i + (length<0)                             
            d2 = np.dot(df2.transpose(),s)

        if success:
            f1 = f2
            print('Cost: ' + str(f1))
            # fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
            s = (np.dot(df2.transpose(),df2)- np.dot(df1.transpose(),df2))/((np.dot(df1.transpose(),df1))*s) - df2;  # Polack-Ribiere direction
        
            #  swap derivatives
            tmp = df1
            df1 = df2
            df2 = tmp                         
            d2 = np.dot(df1.transpose(),s)
            if d2 > 0:                                      # new slope must be negative
                s = -df1                              # otherwise use steepest direction
                d2 = -1.*np.dot(s.transpose(),s)    

            z1 = z1 * min(RATIO, (d1/(d2- 1e-10))[0,0])          # slope ratio but max RATIO
            d1 = d2
            ls_failed = 0                              # this line search did not fail
        else:
            # restore point from before failed line search
            X = X0
            f1 = f0
            df1 = df0  
            if ls_failed | i > abs(length):         # line search failed twice in a row
                break                          # or we ran out of time, so we give up
            # swap derivatives
            tmp = df1
            df1 = df2
            df2 = tmp
            
            s = -df1               # try steepest                                     
            d1 = -1.0 * np.dot(s.transpose(),s)
            z1 = 1./(1.-d1)                   
            ls_failed = 1                                    # this line search failed
              
    return X
