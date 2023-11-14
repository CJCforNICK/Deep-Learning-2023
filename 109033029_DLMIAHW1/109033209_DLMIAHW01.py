import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.linspace(-10, 10, 5000)
# print(x)
# Q1 sigmoid function


def sig(x):
    return 1/(1 + np.exp(-x))


p1_1 = sig(x)
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.plot(x, p1_1, label='Sigmoid(x)')
plt.legend(loc='upper left')
plt.show()
p1_2 = derivative(sig, x)
plt.xlabel('x')
plt.ylabel('The first derivative of Sigmoid(x)')
plt.plot(x, p1_2, label='The first derivative of Sigmoid(x)')
# plt.title('Q1.Sigmoid or logistic function')
plt.legend(loc='upper left')
plt.show()
# Q2 Tanh function


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


p2_1 = tanh(x)
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.plot(x, p2_1, label='Tanh(x)')
plt.legend(loc='upper left')
plt.show()
p2_2 = derivative(tanh, x)
plt.xlabel('x')
plt.ylabel('The first derivative of Tanh(x)')
plt.plot(x, p2_2, label='The first derivative of Tanh(x)')
# plt.title('Q2.Tanh (hyperbolic tangent) function')
plt.legend(loc='upper left')
plt.show()

# Q3 ReLU function check derivative


def relu(x):
    return np.maximum(0.0, x)


def deriv_relu(x):
    data3_2 = [1 if value > 0 else 0 for value in x]
    return np.array(data3_2)


p3_1 = relu(x)
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.plot(x, p3_1, label='ReLU(x)')
plt.legend(loc='upper left')
plt.show()
p3_2 = deriv_relu(x)
plt.xlabel('x')
plt.ylabel('The first derivative of relu(x)')
plt.plot(x, p3_2, label='The first derivative of ReLU(x)')
# plt.title('Q3.ReLU function: (Rectified Linear Unit)')
plt.legend(loc='upper left')
plt.show()

# Q4 Leaky ReLU function


def l_relu(x):
    return np.maximum(0.1*x, x)


def deriv_l_relu(x):
    data4_2 = [1 if value > 0 else 0.1 for value in x]
    return np.array(data4_2)


p4_1 = l_relu(x)
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.plot(x, p4_1, label='leaky_ReLU(x)')
plt.legend(loc='upper left')
plt.show()
p4_2 = deriv_l_relu(x)
plt.xlabel('x')
plt.ylabel('The first derivative of leaky_relu(x)')
plt.plot(x, p4_2, label='The first derivative of leaky_ReLU(x)')
plt.legend(loc='upper left')
# plt.title('Q4.Leaky ReLU function')
plt.show()
# Q5 Parametric ReLU functions


def p_relu(alpha, x):
    return np.maximum(alpha*x, x)


def deriv_p_relu(alpha, x):
    data5_2 = [1 if value > 0 else alpha for value in x]
    return np.array(data5_2)


print('Please enter your parameter alpha for Q5')
alpha = float(input())
print('Parameter input =', alpha)
p5_1 = p_relu(alpha, x)
plt.xlabel('x')
plt.ylabel('parameter_relu(x)')
plt.plot(x, p5_1, label='parameter_ReLU(x)')
plt.legend(loc='upper left')
plt.show()
p5_2 = deriv_p_relu(alpha, x)
plt.xlabel('x')
plt.ylabel('The first derivative of parameter_relu(x)')
plt.plot(x, p5_2, label='The first derivative of parameter_ReLU(x)')
plt.legend(loc='upper left')
# plt.title('Q5.Parametric ReLU function')
plt.show()

# Q6 Exponential Linear Units (ELUs) function


def elus(alpha, x):
    return (x >= 0.0) * x + (x < 0.0) * alpha * (np.exp(x) - 1.0)


def deriv_elus(alpha, x):
    return (x >= 0.0) * 1 + (x < 0.0) * alpha * np.exp(x)


print('Please enter your parameter alpha for Q6')
alpha = float(input())
print('Parameter input =', alpha)
p6_1 = elus(alpha, x)
plt.xlabel('x')
plt.ylabel('ELUs(x)')
plt.plot(x, p6_1, label='ELUs(x)')
plt.legend(loc='upper left')
plt.show()
p6_2 = deriv_elus(alpha, x)
plt.xlabel('x')
plt.ylabel('The first derivative of ELUs(x)')
plt.plot(x, p6_2, label='The first derivative of ELUs(x)')
plt.legend(loc='upper left')
# plt.title('Q6.Exponential Linear Units (ELUs) function')
plt.show()

# Q7 Swish function


def swish(x):
    return (x/(1 + np.exp(-x)))


p7_1 = swish(x)
plt.xlabel('x')
plt.ylabel('swish(x)')
plt.plot(x, p7_1, label='Swish(x)')
plt.legend(loc='upper left')
plt.show()
p7_2 = derivative(swish, x)
plt.xlabel('x')
plt.ylabel('The first derivative of swish(x)')
plt.plot(x, p7_2, label='The first derivative of Swish(x)')
# plt.title('Q7.Swish function')
plt.legend(loc='upper left')
plt.show()

# Q8 Gaussian Error Linear Unit (GELU) function


def gelu(x):
    value = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * value


p8_1 = gelu(x)
plt.xlabel('x')
plt.ylabel('gelu(x)')
plt.plot(x, p8_1, label='GELU(x)')
plt.legend(loc='upper left')
plt.show()
p8_2 = derivative(gelu, x)
plt.xlabel('x')
plt.ylabel('The first derivative of gelu(x)')
plt.plot(x, p8_2, label='The first derivative of GELU(x)')
# plt.title('Q8.Gaussian Error Linear Unit (GELU) function')
plt.legend(loc='upper left')
plt.show()

# Q9 Scaled Exponential Linear Unit (SELU) function


def selus(lambda1, alpha, x):
    return (x >= 0.0) * x * lambda1 + (x < 0.0) * lambda1 * alpha * (np.exp(x) - 1.0)


def deriv_selus(lambda1, alpha, x):
    return (x >= 0.0) * lambda1 + (x < 0.0) * alpha * lambda1 * np.exp(x)


print('Please enter your parameter lambda1 & alpha for Q9')
lambda1, alpha = [float(s) for s in input().split()]
print('Parameter input lambda1 =', lambda1)
print('Parameter input alpha =', alpha)
p9_1 = selus(lambda1, alpha, x)
plt.xlabel('x')
plt.ylabel('selus(x)')
plt.plot(x, p9_1, label='SELUs(x)')
plt.legend(loc='upper left')
plt.show()
p9_2 = deriv_selus(lambda1, alpha, x)
plt.xlabel('x')
plt.ylabel('The first derivative of selus(x)')
plt.plot(x, p9_2, label='The first derivative of SELUs(x)')
plt.legend(loc='upper left')
# plt.title('Q9.Scaled Exponential Linear Unit (SELU) function')
plt.show()
