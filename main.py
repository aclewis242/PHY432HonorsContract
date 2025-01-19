import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score as r2

L = 2.43e-12  # Compton wavelength of electron (in m)
A = 1/137.04  # Fine structure constant
H = 4.136e-21 # Planck's constant (in MeV/Hz)
C = 3e8       # Speed of light (in m/s)

def compton(lam: float, theta: np.ndarray[float]) -> np.ndarray[float]:
    '''
    Compton's equation. Gives scattered wavelength from initial wavelength + scattering angle.

    ### Parameters
    - `lam`: Initial wavelength (m)
    - `theta`: Scattering angle (rad). Accepts either a float or a NumPy array of floats
    '''
    return L*(1 - np.cos(theta)) + lam

def pdf(lam: float, theta: np.ndarray[float]) -> np.ndarray[float]:
    '''
    The Klein-Nishina formula, which serves as a probability density function in this case.

    ### Parameters
    - `lam`: Initial wavelength (m)
    - `theta`: Scattering angle (rad). Accepts either a float or a NumPy array of floats
    '''
    coeff = (A**2*L**2)/(8*np.pi**2)
    r = lam/compton(lam, theta)
    term2 = r + 1/r - np.sin(theta)**2
    return coeff*r**2*term2

def genRandAngle(lam: float, thetas=np.linspace(0, np.pi, 1000)) -> float:
    '''
    Generates and returns a random angle according to the Klein-Nishina formula.

    ### Parameters
    - `lam`: Initial wavelength (m)
    - `thetas`: Scattering angles (rad)
    '''
    probs = pdf(lam, thetas)
    return random.choices(thetas, probs)[0]

def ELconv(lam_or_e: float):
    '''
    Converts between energy (MeV) and wavelength (m). Either quantity is a viable parameter. Accepts NP arrays.
    '''
    return H*C/lam_or_e

def linReg(thetas: np.ndarray[float], es: np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    '''
    Performs linear regression on the given angles (rad) and energies (MeV).

    ### Returns
    - 2-element array of the line's slope & intercept
    - Linearised thetas (1 - cos(theta), x axis)
    - Linearised energies (1/E, y axis)
    '''
    thetas_lin = 1 - np.cos(thetas)
    energies_lin = 1/es
    return np.polyfit(thetas_lin, energies_lin, 1), thetas_lin, energies_lin

def run(sourceMat: tuple[float, str], photonCount: int=10000, binSize: int=6):
    '''
    Runs the simulation.

    ### Parameters
    - `sourceMat`: A tuple of gamma particle energy (MeV) and source material name
    - `photonCount`: The number of photons to simulate
    - `binSize`: The size of the angle bin (intensity)
    '''

    e = sourceMat[0]
    lam = ELconv(e)
    thetas = np.linspace(0, np.pi, 1000)
    intens = np.zeros_like(thetas)
    for n in range(1, photonCount):
        t = genRandAngle(lam, thetas)
        angInd = np.where(thetas == t)[0][0]
        st = angInd - angInd%binSize
        intens[st:st + binSize] += 1
    es = ELconv(compton(lam, thetas))
    lr, ts_p, es_p = linReg(thetas, es)
    lr_line = lr[0]*ts_p + lr[1]
    mOfE = 1/lr[0] # Electron mass
    
    # ENERGY VS ANGLE
    plt.plot(thetas, es)
    plt.ylabel(r"Photon energy $E$ (MeV)")
    plt.xlabel(r"Angle $\theta$ (rad)")
    plt.savefig("photonEnergy_{}.png".format(sourceMat[1]))
    plt.show()

    # INTENSITY VS ANGLE
    fig, ax1 = plt.subplots()
    ax1.scatter(thetas, intens, label='Photon intensities')
    ax1.set_ylabel("Intensity (# photons)")
    ax1.set_xlabel(r"Angle $\theta$ (rad)")
    ax1.set_ylim(ymin=0)
    ax2 = ax1.twinx()
    probsNoNorm = pdf(lam, thetas)
    probs = probsNoNorm/np.linalg.norm(probsNoNorm)
    rsq = r2(intens/np.linalg.norm(intens), probs)
    ax2.plot(thetas, probs, color='r', label='Klein-Nishina formula ($R^2$: {:.4f})'.format(rsq))
    ax2.set_ylabel("Probability")
    ax2.set_ylim(ymin=0)
    plt.legend()
    plt.savefig("photonIntensity_{}.png".format(sourceMat[1]))
    plt.show()

    # ELECTRON MASS LINEAR REGRESSION
    plt.scatter(ts_p, es_p)
    plt.plot(ts_p, lr_line, label='Electron mass: {:.4f} MeV/c$^2$'.format(mOfE))
    plt.ylabel(r"$\frac{1}{E}$ ($E$: scattered photon energy, MeV)")
    plt.xlabel(r"$1 - \cos(\theta)$ ($\theta$: scattering angle, rad)")
    plt.legend()
    plt.savefig("electronMass_{}.png".format(sourceMat[1]))
    plt.show()

if __name__ == '__main__':
    Cs = (0.6617, "Cs-137") # Energy of gamma particles produced by cesium-137 decay (MeV)
    Ba = (0.3560, "Ba-133") # " "   " "              ...produced by barium-133 decay (MeV)

    [run(mat) for mat in [Cs, Ba]]