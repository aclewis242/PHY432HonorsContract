import numpy as np
import matplotlib.pyplot as plt
import random

L = 2.43e-12  # Compton wavelength of electron (in m)
A = 1/137.04  # Fine structure constant
H = 4.136e-15 # Planck's constant (in eV/Hz)
C = 3e8       # Speed of light (in m/s)

def compton(l, theta): # Compton's equation. Gives scattered wavelength from initial wavelength + scattering angle.
    return L*(1 - np.cos(theta)) + l

def pdf(l, theta): # The Klein-Nishina formula, which serves as a probability density function in this case.
    coeff = (A**2*L**2)/(8*np.pi**2)
    r = l/compton(l, theta)
    term2 = r + 1/r - np.sin(theta)**2
    return coeff*r**2*term2

def genRandAngle(l, thetas=np.linspace(0, np.pi, 1000)): # Generates and returns a random angle according to the above PDF.
    probs = pdf(l, thetas)
    return random.choices(thetas, probs)

def ELconv(le): # Converts between energy and wavelength.
    return H*C/le

if __name__ == "__main__":
    e = 0.6617e6 # Energy of gamma particles produced by cesium-137 decay
    l = ELconv(e)
    thetas = np.linspace(0, np.pi, 1000)
    binSize = 6 # To make the intensity data more useful, bins are used
    intens = np.zeros_like(thetas)
    photonCount = 10000
    for n in range(1, photonCount):
        t = genRandAngle(l, thetas)
        angInd = np.where(thetas == t)[0]
        st = angInd - angInd%binSize
        st = st[0]
        intens[st:st + binSize] += 1
    
    plt.plot(thetas, ELconv(compton(l, thetas))*1e-6)
    plt.ylabel(r"Photon energy $E$ (MeV)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    plt.title("Post-collision photon energy as a function of Compton scattering angle")
    plt.show()

    plt.scatter(thetas, intens)
    plt.ylabel(r"Intensity (# photons)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    plt.title("Post-scattering intensity as a function of Compton scattering angle")
    plt.show()

    # Perhaps also produce graphs for 10, 20, ..., 90 degree angles with intensity vs. energy (i.e. like with the lab)?