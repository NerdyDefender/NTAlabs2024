import sympy
import math
import random
import numpy as np
import threading
import time

def check_b_smooth(fbase, number_to_check):
    power_factor_vector = [0] * len(fbase)
    while number_to_check > 1:
        for factor in fbase:
            if number_to_check % factor == 0:
                power_factor_vector[fbase.index(factor)] += 1
                number_to_check //= factor
                break
        else:
            return False
    return [i for i in power_factor_vector]


def generate_half_equations(target_length, results):
    equations = []
    while len(equations) < target_length:
        k = random.randint(a=1, b=n - 1)
        power_vector = check_b_smooth(fbase=factor_base, number_to_check=pow(base=alpha, exp=k, mod=modulo))
        if power_vector is not False:
            power_vector.append(k)
            equations.append(power_vector)
    results.extend(equations)


def generate_equations():
    total_length = len(factor_base) + const
    half_length = total_length // 2

    results1 = []
    results2 = []

    thread1 = threading.Thread(target=generate_half_equations, args=(half_length, results1))
    thread2 = threading.Thread(target=generate_half_equations, args=(total_length - half_length, results2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    return results1 + results2


def is_row_echelon_form(matrix):
    if not matrix.any():
        return False

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    prev_leading_col = -1

    for row in range(rows):
        leading_col_found = False
        for col in range(cols):
            if matrix[row, col] != 0:
                if col <= prev_leading_col:
                    return False
                prev_leading_col = col
                leading_col_found = True
                break
        if not leading_col_found and any(matrix[row, col] != 0 for col in range(cols)):
            return False
    return True


def find_nonzero_row(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    for row in range(pivot_row, nrows):
        if matrix[row, col] != 0:
            return row
    return None


def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]


def make_pivot_one(matrix, pivot_row, col):
    pivot_element = matrix[pivot_row, col]
    matrix[pivot_row] //= pivot_element


def eliminate_below(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    pivot_element = matrix[pivot_row, col]
    for row in range(pivot_row + 1, nrows):
        factor = matrix[row, col]
        matrix[row] -= factor * matrix[pivot_row]


def row_echelon_form(matrix):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    pivot_row = 0
    for col in range(ncols):
        nonzero_row = find_nonzero_row(matrix, pivot_row, col)
        if nonzero_row is not None:
            swap_rows(matrix, pivot_row, nonzero_row)
            make_pivot_one(matrix, pivot_row, col)
            eliminate_below(matrix, pivot_row, col)
            pivot_row += 1
    return matrix


def find_x():
    while True:
        k = random.randint(a=1, b=n - 1)
        beta_mul_alpha_pow_k = (beta * pow(base=alpha, exp=k, mod=modulo)) % modulo
        power_vector = check_b_smooth(fbase=factor_base, number_to_check=beta_mul_alpha_pow_k)
        if power_vector is not False:
            unknown_x = (sum(ai * bi for ai, bi in zip(power_vector, answer)) - k) % n
            return unknown_x
        else:
            continue


def findind_small_logarithms():
    true_answer = [0] * len(factor_base)
    while 0 in true_answer:
        try:
            equations = generate_equations()
            matrix_equation = np.array(equations)
            result = row_echelon_form(matrix_equation)

            lst = result[:len(factor_base)].tolist()
            res = []
            for j in lst:
                res.append(j.pop())

            A = sympy.Matrix(lst)
            B = sympy.Matrix(res)
            det = int(A.det())
            if math.gcd(det, n) == 1:
                not_true_answer = list(pow(det, -1, n) * A.adjugate() @ B % n)

            for j in range(len(factor_base)):
                number = pow(base=alpha, exp=int(not_true_answer[j]), mod=modulo)
                if number == factor_base[j] and true_answer[j] == 0:
                    true_answer[j] = int(not_true_answer[j])
                    print(true_answer)
        except UnboundLocalError as e:
            continue
    return true_answer



alpha = int(input("Please enter alpha: "))
beta = int(input("Please enter beta: "))
modulo = int(input("Please enter p: "))
start = time.time()
n = modulo - 1
const = 10
exp = 1 / 2 * math.sqrt(math.log(n) * math.log(math.log(n)))
b = 3.38 * pow(math.e, exp=exp)
factor_base = [prime for prime in sympy.primerange(3, int(b))]


answer = findind_small_logarithms()
x = int(find_x())
print(f"Answer to equation {alpha}^x = {beta} mod {modulo} is {x}")
print(f"Time taken {time.time() - start:.3f}")
