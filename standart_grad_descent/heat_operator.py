import numpy as np

# прямой оператор

class heat_operator:
    def __init__(self, a, T, x):
        self.a = a
        self.T = T
        self.x = x
    
    def __call__(self, u):
        return self.heat_forward(u, self.a, self.T, self.x)
    
    def heat_forward(self, u, a, T, x):
        N = len(x)
        L = np.max(np.abs(x))

        k = (2 * np.pi / (2 * L)) * np.concatenate((
            np.arange(0, N // 2),
            np.arange(-N // 2, 0)
        ))

        u_hat = np.fft.fft(u)
        G_hat = np.exp(-a * T * k**2)
        y_hat = G_hat * u_hat

        return np.real(np.fft.ifft(y_hat))

    
