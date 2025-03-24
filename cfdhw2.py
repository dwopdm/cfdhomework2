import sys
import numpy as np
import matplotlib.pyplot as plt

# 确保环境支持 numpy 和 matplotlib
try:
    import micropip
    micropip.install(["numpy", "matplotlib"])
except ModuleNotFoundError:
    print("micropip not available, ensure numpy and matplotlib are installed.", file=sys.stderr)

# 定义有限差分格式
class FiniteDifference:
    def __init__(self, dx):
        self.dx = dx
    
    def first_derivative_forward(self, u):
        """前向差分格式 O(dx) (模板: u[i], u[i+1])"""
        return (u[1:] - u[:-1]) / self.dx
    
    def first_derivative_central(self, u):
        """中心差分格式 O(dx^2) (模板: u[i-1], u[i], u[i+1])"""
        return (u[2:] - u[:-2]) / (2 * self.dx)
    
    def second_derivative_central(self, u):
        """中心差分格式 O(dx^2) (模板: u[i-1], u[i], u[i+1])"""
        return (u[:-2] - 2 * u[1:-1] + u[2:]) / (self.dx ** 2)
    
    def second_derivative_forward(self, u):
        """前向差分格式 O(dx) (模板: u[i], u[i+1], u[i+2], u[i+3])
        注意：该格式仅适用于靠近边界的点，其导数数组长度为 len(u)-3
        """
        return (2*u[:-3] - 5*u[1:-2] + 4*u[2:-1] - u[3:]) / (self.dx**2)

    
    

# 数值验证误差
class ErrorAnalysis:
    def __init__(self, dx, x_range):
        self.dx = dx
        self.x = np.arange(x_range[0], x_range[1], dx)
    
    def test_polynomial(self):
        """测试多项式函数，如 u(x) = x^3"""
        u = self.x**3
        exact_first = 3 * self.x**2
        exact_second = 6 * self.x
        return u, exact_first, exact_second
    
    def test_sine(self, k):
        """测试正弦函数 u(x) = sin(kx)"""
        u = np.sin(k * self.x)
        exact_first = k * np.cos(k * self.x)
        exact_second = -k**2 * np.sin(k * self.x)
        return u, exact_first, exact_second

# 计算误差并绘图
class PrecisionComparison:
    @staticmethod
    def compute_error(numerical, exact):
        return np.abs(numerical - exact).max()
    
    @staticmethod
    def plot_results(x, numerical, exact, title):
        plt.figure(figsize=(8, 5))
        plt.plot(x, numerical, 'r--', label='Numerical')
        plt.plot(x, exact, 'b-', label='Exact')
        plt.legend()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Value')
        plt.show()

# 运行测试
if __name__ == "__main__":
    dx = 0.01
    x_range = (0, 2 * np.pi)
    fd = FiniteDifference(dx)
    analysis = ErrorAnalysis(dx, x_range)
    
    # 选择测试函数（多项式或正弦函数）
    u, exact_first, exact_second = analysis.test_sine(k=1)
    
    # 计算数值导数
    numerical_first = fd.first_derivative_central(u)
    numerical_first_forward = fd.first_derivative_forward(u)
    numerical_second = fd.second_derivative_central(u)
    numerical_second_forward = fd.second_derivative_forward(u)

    
    # 计算误差
    error_first = PrecisionComparison.compute_error(numerical_first, exact_first[1:-1])
    error_first_forward = PrecisionComparison.compute_error(numerical_first_forward, exact_first[:len(numerical_first_forward)])
    error_second = PrecisionComparison.compute_error(numerical_second, exact_second[1:-1])
    
    error_second_forward = PrecisionComparison.compute_error(numerical_second_forward, exact_second[:len(numerical_second_forward)])
    
    # 画图
    PrecisionComparison.plot_results(analysis.x[1:-1], numerical_first, exact_first[1:-1], 'First Derivative')
    PrecisionComparison.plot_results(analysis.x[1:-1], numerical_second, exact_second[1:-1], 'Second Derivative')
    PrecisionComparison.plot_results(analysis.x[:len(numerical_second_forward)], numerical_second_forward, exact_second[:len(numerical_second_forward)], 'Second Derivative (Forward)')
    PrecisionComparison.plot_results(analysis.x[:len(numerical_first_forward)], numerical_first_forward, exact_first[:len(numerical_first_forward)], 'First Derivative (Forward)')
    print(f'First derivative error: {error_first}')
    print(f'Second derivative error: {error_second}')
    print(f'First derivative forward error: {error_first_forward}')
    print(f'Second derivative forward error: {error_second_forward}')