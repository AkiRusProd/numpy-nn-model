


class Adam:
    def __init__(self, params, lr: float=0.01, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]
        self.v = [param.xp.zeros_like(param.data) for param in self.params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.grad**2

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (param.xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class SGD:
    def __init__(self, params, lr: float=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = None

class Momentum:
    def __init__(self, params, lr=0.01, momentum: float=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.m = [param.xp.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * param.grad
            param.data -= self.m[i] * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = None if param.grad is None else param.xp.zeros_like(param.grad)


class RMSprop:
    def __init__(self, params, lr: float=0.01, alpha: float=0.99, eps: float=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.alpha * self.m[i] + (1 - self.alpha) * param.grad**2
            param.data -= self.lr * param.grad / (param.xp.sqrt(self.m[i]) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class Adagrad:
    def __init__(self, params, lr: float=0.01, eps: float=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] += param.grad**2
            param.data -= self.lr * param.grad / (param.xp.sqrt(self.m[i]) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class Adadelta:
    def __init__(self, params, lr: float=1.0, rho: float=0.9, eps: float=1e-6):
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]
        self.v = [param.xp.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.rho * self.m[i] + (1 - self.rho) * param.grad**2
            delta_x = (
                -(param.xp.sqrt(self.v[i] + self.eps) / param.xp.sqrt(self.m[i] + self.eps))
                * param.grad
            )
            self.v[i] = self.rho * self.v[i] + (1 - self.rho) * delta_x**2
            param.data += delta_x

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class Adamax:
    def __init__(self, params, lr: float=0.002, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]
        self.v = [param.xp.zeros_like(param.data) for param in self.params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = param.xp.maximum(self.betas[1] * self.v[i], param.xp.abs(param.grad))

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)

            param.data -= self.lr * m_hat / (self.v[i] + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class NAdam:
    def __init__(self, params, lr: float=0.002, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.m = [param.xp.zeros_like(param.data) for param in self.params]
        self.v = [param.xp.zeros_like(param.data) for param in self.params]

        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.grad**2

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t) + (1 - self.betas[0]) * param.grad / (
                1 - self.betas[0] ** self.t
            )
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (param.xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None
