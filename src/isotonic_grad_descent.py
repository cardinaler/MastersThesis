import numpy as np


def pava(y):
    """Проекция на конус монотонно невозрастающих функций."""
    n = y.shape[0]
    u = y.copy().astype(float)
    blocks = [[i, 1, u[i]] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i+1][2]:
            start, n1, val1 = blocks[i]
            _, n2, val2 = blocks[i+1]
            new_val = (n1 * val1 + n2 * val2) / (n1 + n2)
            blocks[i] = [start, n1 + n2, new_val]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    for start, count, val in blocks:
        u[start: start + count] = val
    return u


class isotonic_grad_descent:
    """Класс для выполнения градиентного спуска с проекцией на каждом шаге

    Args:
        alpha (float): Шаг градиентного спуска
        u_init (np.ndarray): Начальный вектор значений.
        A (np.ndarray): Симметричная матрица оператора (A^T = A).
        g (np.ndarray): Вектор свободных членов (правая часть).
        tau (float): Параметр регуляризации или пороговое значение.
        proj_type (str или Callable): Метод проекции на каждом шаге.
            Ожидает 'pava', 'non_negative'
    """

    def __init__(self, alpha, u_init, A, g, tau, proj_type):
        self.alpha = alpha
        self.u_init = u_init
        self.A = A  # Здесь A_T = A
        self.g = g
        self.tau = tau
        self.proj_type = proj_type

    def start(self, max_iter=500000):
        u = self.u_init.copy()
        # Начальное значение функции потерь
        Au = self.A(u)
        loss = 0.5 * np.linalg.norm(Au - self.g)**2 + \
            self.alpha * 0.5 * np.linalg.norm(u)**2

        for it in range(1, max_iter + 1):
            # Пересчитываем Au для текущего u
            Au = self.A(u)

            # Градиент: A(Au - g) + alpha * u
            grad = self.A(Au - self.g) + self.alpha * u

            # Градиентный шаг
            u = u - self.tau * grad
            # найти минимальное значение, отсечь все отсальные а затем после проекции отразить
            # Проекция на монотонные функции (PAVA)
#            u = np.concatenate((u[::-1], u))
            if self.proj_type == 'non_negative':
                u = np.maximum(u, 0)
            elif self.proj_type == 'pava':
                n = len(u)
                half_len = n // 2

                # Берем первую половину
                u = u[:half_len]
                u = pava(u)
                if n % 2 == 0:
                    # Для ЧЕТНОЙ длины (например, 6 элементов -> 3 и 3)
                    u = np.concatenate((u, np.flip(u)))
                else:
                    center = u[half_len: half_len + 1]
                    u = np.concatenate((u, center, np.flip(u)))

            # Проверка сходимости каждые 1000 итераций
            if it % 1000 == 0:
                current_Au = self.A(u)
                loss1 = 0.5 * \
                    np.linalg.norm(current_Au - self.g)**2 + \
                    self.alpha * 0.5 * np.linalg.norm(u)**2

                if abs(loss - loss1) < 1e-8:
                    print(f"Converged at iter {it:5d}, loss = {loss1:.6e}")
                    break

                loss = loss1
                print(f"iter = {it:5d}, loss = {loss:.6e}")

        return u
