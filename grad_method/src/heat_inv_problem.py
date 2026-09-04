import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from src.isotonic_grad_descent import isotonic_grad_descent
from src.heat_operator import heat_operator

class heat_inv_problem:
    def __init__(self, a, x):
        self.a = a
        self.x = x

    def create_task(self, T, u_true): # A(u_true) = g
        self.A = heat_operator(self.a, T, self.x)
        self.g = self.A(u_true)
        self.T = T
        self.u_true = u_true

        return self.g

    def solve_task(self, alpha, tau, proj_type):
        u_init = np.zeros_like(self.x)
        solver = isotonic_grad_descent(alpha, u_init, self.A, self.g, tau, proj_type)
        self.u = solver.start()


    def visualize_task(self, x_lim, y_lim, name):
        # визуализация
        
        plt.figure(figsize=(10, 7))

        plt.plot(self.x, self.u_true, 'k', linewidth=3, label='True f(x)')
        plt.plot(self.x, self.u, 'r--', linewidth=2, label='Restored f(x)')

        plt.xlim(x_lim)
        plt.ylim(y_lim)

        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('f')
        plt.title(f'График восстановленной и истинной f(x), полученой из $u_t(x, {self.T})$')
        plt.legend()
        plt.savefig(f"{name}.pdf", bbox_inches='tight') # 'bbox_inches="tight"' removes extra whitespace
        plt.show()

    def integrate_heat_operator(self, u0, t_final):
        """
        Интегрирует ваш оператор по времени от 0 до t_final с помощью SciPy.
        """
        # Инициализация класса
        op = heat_operator(self.a, T=0, x = self.x)
        
        # Подынтегральная функция-обертка.
        # quad_vec ожидает, что функция принимает первым аргументом переменную интегрирования, то есть y
        def integrand(y):
            return op.heat_forward(u0, a = self.a, T=y, x=self.x)
        
        # Вызываем готовый метод интегрирования от 0 до t_final
        solution, error = quad_vec(integrand, 0, t_final)
        
        return solution

