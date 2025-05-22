from sympy import symbols
from sympy.utilities.lambdify import lambdify
import sympy as sp

def case_1():
    # Define symbols
    x, y = symbols('x y')

    # Define a symbolic expression
    expr = x**2 + y**2

    # Convert the symbolic expression into a Python function
    f = lambdify((x, y), expr)

    # Now you can evaluate the function numerically
    result = f(2, 3)  # This will evaluate x^2 + y^2 at x=2, y=3
    print(result)  # Output will be 13 (2^2 + 3^2)


def case_2():
    # Define matrices
    x, y = symbols('x y')
    relative_state = sp.Matrix([[x], [y]])  # 2x1 matrix
    print('relative_state shape:', relative_state.shape)  # This will output (2, 1)

    H = sp.Matrix([[1.0, 0.0], [0.0, 1.0]])  # 2x2 identity matrix

    # Matrix multiplication (will return a 1x1 matrix)
    result = relative_state.T @ H @ relative_state

    # Output of result is a 1x1 matrix
    print(result)  # This will output a 1x1 matrix like Matrix([[5]])

    # Accessing the scalar value inside the matrix
    scalar_value = result[0, 0]
    print(scalar_value)  # This will output 5 as a scalar



if __name__ == '__main__':
    # case_1()
    case_2()