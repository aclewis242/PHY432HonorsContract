import numpy as np
import matplotlib.pyplot as plt
import random

L = 2.43e-12 # Compton wavelength of electron
A = 1/137.04 # Fine structure constant

def ratio(l, theta):
    eps = L/l
    return 1/(1 + eps*(1 - np.cos(theta)))

def pdf(l, theta):
    coeff = (A**2*L**2)/(8*np.pi**2)
    r = ratio(l, theta)
    term2 = r + 1/r - np.sin(theta)**2
    return coeff*r**2*term2

def genRandAngle(l, tmin=0, tmax=np.pi):
    thetas = np.linspace(tmin, tmax, 1000)
    probs = pdf(l, thetas)
    return random.choices(thetas, probs)