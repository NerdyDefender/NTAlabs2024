import random
import math
import time
import sympy
import random


def miller_rabin_test(p, k=30):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
              227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    if p % 2 == 0:
        return False

    # попереднє ділення
    for num in primes:
        if p % num == 0:
            return False

    # знаходимо розклад p-1 = d * 2^s
    s = 0
    d = p - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for i in range(1, k + 1):
        x = random.randint(1, p)
        gcd = math.gcd(x, p)
        x = pow(x, d, p)
        # print(x)
        if gcd > 1:
            return False
        if x == 1 or x == p - 1:
            continue

        for r in range(s - 1):
            x = pow(x, 2, p)
            if x == p - 1:
                break
            elif x % p == 1:
                return False
        else:
            return False

    return True


def trial_division(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    while n % 3 == 0:
        factors.append(3)
        n //= 3
    i = 5
    while i <= 47:  
        while n % i == 0:
            factors.append(i)
            n //= i
        while n % (i + 2) == 0:
            factors.append(i + 2)
            n //= i + 2
        i += 6
    return factors


def pollard_rho_2k(n):
    if n % 2 == 0:
        return 2

    for _ in range(1000):
        f = lambda x: pow(x, 2) - 1
        x = y = 2
        d = 1
        while d == 1:
            x = f(x) % n
            y = f(f(y)) % n
            d = math.gcd((x - y) % n, n)
        if d != n:
            return d


def check_b_smooth(fbase, number_to_check):
    power_factor_vector = [0] * len(fbase)
    if number_to_check <= -1:
        number_to_check = abs(number_to_check)
        power_factor_vector[0] += 1
    while number_to_check > 1:
        for factor in fbase[1:]:
            if number_to_check % factor == 0:
                power_factor_vector[fbase.index(factor)] += 1
                number_to_check //= factor
                break
        else:
            return False
    return [i % 2 for i in power_factor_vector]


def generate_k_plus_one_bsmooth(num, fbase):
    bsmooth = []
    vectors = []
    v = [1]
    alpha = math.sqrt(num)
    a = [int(alpha)]
    u = [a[0]]
    b = [0, 1]  # початкові значення b_0 і b_1
    while len(bsmooth) < len(fbase):
        v.append((num - u[-1] ** 2) // v[-1])
        b_i = (a[-1] * b[-1] + b[-2]) % num
        b.append(b_i)
        a.append(int((alpha + u[-1]) / v[-1]))
        u.append(a[-1] * v[-1] - u[-1])
        bsquare = pow(b[-1], 2) % num
        vector = check_b_smooth(number_to_check=bsquare, fbase=fbase)
        if vector is not False:
            bsmooth.append((b[-1], bsquare % num))
            vectors.append(vector)
    return bsmooth, vectors


def row_reduce(matrix) -> list:
    # транспонуємо матрицю
    matrix_t = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    for j in range(len(matrix_t)):
        index = max(range(0, len(matrix_t[j])), key=lambda x: matrix_t[j][x])
        if matrix_t[j][index] == 0:
            continue
        for k in range(len(matrix_t)):
            if k == j:
                continue
            if matrix_t[k][index] == 1:
                matrix_t[k] = [(matrix_t[k][m] + matrix_t[j][m]) % 2 for m in range(len(matrix_t[j]))]

    # транспонуємо назад
    matrix_t = [[matrix_t[j][i] for j in range(len(matrix_t))] for i in range(len(matrix_t[0]))]

    return matrix_t


def solve_sle_gf2(matrix: list) -> list:
    matrix = row_reduce(matrix)
    full_solution = []
    # знаходимо всі нульові вектори
    zero_vectors = [i for i, x in enumerate(matrix) if x == [0] * len(x)]
    for v in zero_vectors:
        full_solution.append({v})

    for i in range(len(matrix)):
        i_th_row = matrix[i]
        # індекси i-того рядка де елементи = 1
        indexes_of_1_in_ith_row = [x for x in range(len(i_th_row)) if i_th_row[x] == 1]

        if sum(matrix[i]) >= 1:
            partial_solution = {i}
            # шукаємо такі рядки,де на місці одиниці і-ого рядка також стоїть одиниця
            for k in indexes_of_1_in_ith_row:
                # проходимо по всіх рядках матриці
                for j in range(len(matrix)):
                    # пропускаємо рядок,якщо внутрішній цикл натрапив на той самий рядок,що і зовнішній
                    if i == j:
                        continue
                    # шукаємо рядки матриці,такі щоб дорівнювали і-ому рядку (x xor x = 0)
                    if matrix[i] == matrix[j]:
                        identical_vectors = {i, j}
                        if identical_vectors not in full_solution:
                            full_solution.append(identical_vectors)
                        continue
                    # перша умова перевіряє чи рядок детермінований,а друга чи 1 стоїть на потрібному місці
                    if matrix[j].count(1) == 1 and matrix[j][k] == 1:
                        partial_solution.add(j)
                        i_th_row = [(matrix[j][idx] + i_th_row[idx]) % 2 for idx in range(len(i_th_row))]
                        if sum(i_th_row) == 0:
                            if partial_solution not in full_solution:
                                full_solution.append(partial_solution)
                        break
    return full_solution


def brilhart_morrison_factorization(number):
    a = [1/math.sqrt(2),1,math.sqrt(3)/math.sqrt(2),math.sqrt(2)]
    for i in a:
        factor_base = [-1]
        L = sympy.exp((sympy.log(number) * sympy.log(sympy.log(number))) ** 0.5)
        primes = [p for p in sympy.primerange(3, int(L)) if 1 < p < L ** i and sympy.legendre_symbol(number, p) == 1]
        factor_base.extend(primes)
        b_smooth, power_vectors = generate_k_plus_one_bsmooth(num=number, fbase=factor_base)
        solution = solve_sle_gf2(power_vectors)
        factors = []
        for s in solution:
            X = 1
            Y = 1
            for i in s:
                X *= b_smooth[i][0]
                Y *= b_smooth[i][1]
            X = X % number
            Y = Y % number
            gcd = math.gcd(X - int(math.sqrt(Y)), number)
            if gcd != 1 and gcd != number:
                factors.append(gcd)
            else:
                continue
            return factors

# ---------------------------------------------------------------------------------------------------------------------
# Основна програма
# variant_number = 691534156424661573
# start_time = time.time()
# print(f"Program started,factored number = {variant_number}")
# while True:
#     if miller_rabin_test(variant_number) is True:
#         variant_factors.append(variant_number)
#         break
#     found = trial_division(variant_number)
#     if len(found) != 0:
#         print(f"Trivial division found factors {str(found)}, time passed from start: {time.time() - start_time}")
#         variant_factors.extend(found)
#         for factor in found:
#             variant_number //= factor
#         continue
#     found = pollard_rho_2k(n=variant_number)
#     if found is not None:
#         print(f"Pollard rho algorithm found factors {found}, time passed from start: {time.time() - start_time}")
#         variant_factors.append(found)
#         variant_number //= found
#
#     found = brilhart_morrison_factorization(number=variant_number)
#     if found is not None:
#         print(f"Brilhart morrison algorithm found factors {found}, time passed from start: {time.time()-start_time}")
#         variant_factors.extend(found)
#         for factor in found:
#             variant_number //= factor
# #
# print(f"Variant factors: {variant_factors}")

# ---------------------------------------------------------------------------------------------------------------------
# Порівнюємо алгоритми
# nums = [3009182572376191, 1021514194991569, 4000852962116741, 15196946347083, 499664789704823, 269322119833303,
#         679321846483919, 96267366284849, 61333127792637, 2485021628404193]
#
#
# def compare(number):
#     print(f"Factoring {number}")
#     start_time = time.time()
#     factors_brilhart = brilhart_morrison_factorization(number)
#     brilhart_time = time.time()
#     print(f"Brilhart morisson found factor {factors_brilhart} in {brilhart_time - start_time} seconds")
#     pollard_factors = pollard_rho_2k(number)
#     pollard_time = time.time()
#     print(f"Pollard rho found factor {pollard_factors} in {pollard_time-brilhart_time} seconds")
#     return
#
#
# for num in nums:
#     compare(number=num)
# ---------------------------------------------------------------------------------------------------------------------
# Визначення максимального порядку числа яке піддається факторизації
# for bits in range(40, 256):
#     num = random.getrandbits(bits)
#     start_time = time.time()
#     factors = []
#     print(f"Program started,factored number = {num}, bit length : {bits}")
#     while True:
#         if miller_rabin_test(num) is True:
#             factors.append(num)
#             break
#         found = trial_division(num)
#         if len(found) != 0:
#             print(f"Trivial division found factors {str(found)}, time passed from start: {time.time() - start_time}")
#             factors.extend(found)
#             for factor in found:
#                 num //= factor
#             continue
#         found = pollard_rho_2k(n=num)
#         if found is not None:
#             print(f"Pollard rho algorithm found factors {found}, time passed from start: {time.time() - start_time}")
#             factors.append(found)
#             num //= found
#         found = brilhart_morrison_factorization(number=num)
#         if found is not None:
#             print(f"Brilhart morrison algorithm found factors {found}, time passed from start: {time.time()-start_time}")
#             factors.extend(found)
#             for factor in found:
#                 num //= factor
#     print(f"Factors: {factors}")
