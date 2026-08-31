import numpy as np


class grad_descent:
    # descent for (||Au - g||_2)^2 + alpha (||u||_2)^2
    def __init__(self, alpha, u_init, A, g, tau):
        self.alpha = alpha
        self.u_init = u_init
        self.A = A
        self.g = g
        self.tau = tau
    
    def start(self):
        u = self.u_init
        Au = self.A(u)
        loss = 0.5 * np.linalg.norm(Au - self.g)**2 + self.alpha * 0.5 * np.linalg.norm(u)**2
        it = 0
        while(True):
            it += 1
            Au = self.A(u)

            # градиент невязки
            grad_data = self.A(Au - self.g)

            # субградиент L1
            grad_reg = u

            # общий градиент
            grad = grad_data + self.alpha * grad_reg

            # шаг градиентного спуска
            u = u - self.tau * grad
            u = np.maximum(u, 0)

            if it % 1000 == 0:
                loss1 = 0.5 * np.linalg.norm(Au - self.g)**2 + self.alpha * 0.5 * np.linalg.norm(u)**2
                if abs(loss - loss1) < 1e-8:
                    break
                loss = loss1
                print(f"iter = {it:5d}, loss = {loss:.6e}")

        return u
