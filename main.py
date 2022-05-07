import numpy as np
import matplotlib.pyplot as plt
import random

L = 2.43e-12  # Compton wavelength of electron (in m)
A = 1/137.04  # Fine structure constant
H = 4.136e-21 # Planck's constant (in MeV/Hz)
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

def linReg(ts, es): # Performs linear regression on the given angles and energies.
    thetas_p = 1 - np.cos(ts)
    energies_p = 1/es
    return np.polyfit(thetas_p, energies_p, 1), thetas_p, energies_p

if __name__ == "__main__":
    Cs = (0.6617, "Cs-137") # Energy of gamma particles produced by cesium-137 decay (MeV)
    Ba = (0.3560, "Ba-133") # " "   " " ...produced by barium-133 decay (MeV)

    ### CHANGE THESE LINES TO CHANGE PARAMETERS
    eTup = Ba           # Source material
    photonCount = 10000 # Number of photons
    binSize = 6         # Size of angle bin (for intensity)

    e = eTup[0]
    l = ELconv(e)
    thetas = np.linspace(0, np.pi, 1000)
    intens = np.zeros_like(thetas)
    for n in range(1, photonCount):
        t = genRandAngle(l, thetas)
        angInd = np.where(thetas == t)[0]
        st = angInd - angInd%binSize
        st = st[0]
        intens[st:st + binSize] += 1
    es = ELconv(compton(l, thetas))
    lr, ts_p, es_p = linReg(thetas, es)
    lr_line = lr[0]*ts_p + lr[1]
    mOfE = 1/lr[0]
    
    # ENERGY VS ANGLE
    plt.plot(thetas, es)
    plt.ylabel("Photon energy $E$ (MeV)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    # plt.title("Post-collision photon energy by Compton scattering angle ({})".format(eTup[1]))
    plt.savefig("photonEnergy_{}.png".format(eTup[1]))
    plt.show()

    # INTENSITY VS ANGLE
    plt.scatter(thetas, intens)
    plt.ylabel("Intensity (# photons)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    # plt.title("Post-scattering intensity by Compton scattering angle ({})".format(eTup[1]))
    plt.savefig("photonIntensity_{}.png".format(eTup[1]))
    plt.show()

    ### This section has been deprecated -- there seemed to be some sort of correlation between energy & intensity when intensity was scaled
    ### logarithmically, but after a few trials with different parameter values, I concluded that the two just happened to look similar in
    ### this case
    # ENERGY, INTENSITY OVERLAY
    # fig, ax1 = plt.subplots()
    # ax1.scatter(thetas, intens)
    # ax1.set_ylabel("Intensity (# photons)")
    # ax1.set_yscale('log')
    # ax2 = ax1.twinx()
    # ax2.plot(thetas, es, color='r')
    # ax2.set_ylabel("Photon energy $E$ (MeV)")
    # ax2.set_xlabel(r"Angle $\theta$ (rad)")
    # plt.title("Energy, intensity overlaid")
    # plt.savefig("energyIntensity.png")
    # plt.show()

    # ELECTRON MASS LINEAR REGRESSION
    plt.scatter(ts_p, es_p)
    plt.plot(ts_p, lr_line, label='Electron mass: {:.4f} MeV/c$^2$'.format(mOfE))
    plt.ylabel(r"$\frac{1}{E}$ ($E$: scattered photon energy, MeV)")
    plt.xlabel(r"$1 - \cos(\theta)$ ($\theta$: scattering angle, rad)")
    # plt.title("Mass of electron from data ({})".format(eTup[1]))
    plt.legend()
    plt.savefig("electronMass_{}.png".format(eTup[1]))
    plt.show()