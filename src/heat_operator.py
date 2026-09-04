import numpy as np

# прямой оператор

class heat_operator:
    def __init__(self, a, T, x):
        self.a = a
        self.T = T
        self.x = x
    
    def __call__(self, u):
        return self.heat_forward(u, self.a, self.T, self.x)
    
    def heat_forward(self, u0, a, T, x):
        dx = x[1] - x[0]
        N = len(x)

        k = np.fft.rfftfreq(N, d=dx) * 2 * np.pi

        u_hat = np.fft.rfft(u0)
        G_hat = np.exp(- (a**2) * (k**2) * T)
        y_hat = G_hat * u_hat

        return np.fft.irfft(y_hat, n=N)

    
