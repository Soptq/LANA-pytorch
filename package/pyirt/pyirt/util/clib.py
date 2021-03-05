# -*-coding:utf-8-*-

import numpy as np

exp = np.exp
log = np.log

def log_likelihood_2PL(y1,  y0, theta, alpha, beta, c=0.0):
    expPos = exp(alpha * theta + beta)
    ell =  y1 * log((c + expPos) / (1.0 + expPos)) + y0 * log((1.0 - c) / (1.0 + expPos)) ;

    return ell


def log_likelihood_2PL_gradient(y1, y0, theta, alpha, beta, c=0.0):
    grad = np.zeros(2)

    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    temp = exp(beta + alpha * theta)
    beta_grad = temp / (1.0 + temp) * ( y1 * (1.0 - c) / (c + temp) - y0)

    alpha_grad = theta * beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad
