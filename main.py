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

def linReg(ts, es):
    thetas_p = 1 - np.cos(ts)
    energies_p = 1/es
    return np.polyfit(thetas_p, energies_p, 1), thetas_p, energies_p

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
    es = ELconv(compton(l, thetas))*1e-6
    lr, ts_p, es_p = linReg(thetas, es)
    lr_line = lr[0]*ts_p + lr[1]
    mOfE = 1/lr[0]
    
    plt.plot(thetas, es)
    plt.ylabel(r"Photon energy $E$ (MeV)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    plt.title("Post-collision photon energy as a function of Compton scattering angle")
    plt.show()

    plt.scatter(thetas, intens)
    plt.ylabel(r"Intensity (# photons)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    plt.title("Post-scattering intensity as a function of Compton scattering angle")
    plt.show()

    plt.scatter(ts_p, es_p)
    plt.plot(ts_p, lr_line, label='Electron mass: {:.4f} MeV/c$^2$'.format(mOfE))
    plt.ylabel(r"$\frac{1}{E}$ ($E$: scattered photon energy, MeV)")
    plt.xlabel(r"$1 - \cos(\theta)$ ($\theta$: scattering angle, rad)")
    plt.title("Mass of electron from data")
    plt.legend()
    plt.show()