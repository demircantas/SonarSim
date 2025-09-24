import numpy as np

def eval_u_Sonar(t):
    """
    Sonar ping with Gaussian envelope 
    """
    f0 = 10000         # Hz
    t0 = 0.001         # s pulse center
    sigma = 0.0001     # s pulse width
    A0 = 1e2           # normalized amplitude
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    
    # only significant within 3 sigma of center
    if abs(t - t0) > 3 * sigma:
        return 0.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)
