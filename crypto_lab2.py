import time

def discrete_log(base, result, mod):
    # Шукаємо дискретний логарифм
    x = 1
    start_time = time.time()
    while True:
        if pow(base, x, mod) == result:
            return x, start_time - time.time()
        x += 1
        # Перевіряємо час виконання
        if time.time() - start_time >= 600:  # 5 хвилин
            return x,22

# base = 6
# result = 7531
# mod = 8101
# solution, time_spend = discrete_log(base, result, mod)
# if solution is not None:
#     print(f"Дискретний логарифм для {result} за основою {base} по модулю {mod} дорівнює {solution}.Час {time_spend}")
# else:
#     print("Час вичерпано. Розв'язок не знайдено aбо розв'язок не знайдено")





# def silver_polig_hellman(base, result, modulo):
#     factors = [2, 2, 3, 3, 3, 3, 5, 5]
#     distinct_factors = sorted(list(set(factors)))
#     powers = [factors.count(x) for x in distinct_factors]
#     print("Distinct Factors:", distinct_factors)
#     print("Powers:", powers)
#     r_i_j = [[] for _ in range(len(distinct_factors))]
#     # creating r_ij tables for each p_i
#     n = modulo - 1
#     for i, p_i in enumerate(distinct_factors):
#         for j in range(p_i):
#             r_j = pow(base, n * j // p_i, modulo)
#             r_i_j[i].append(r_j)
#     print("r_i_j:", r_i_j)
#     x_for_each_factor = [[] for x in range(len(distinct_factors))]
#     # проходимо по всіх p_i
#     for i, p_i in enumerate(distinct_factors):
#         # для кожного p_i шукаємо x_i
#         print(f"working with p{i} = {p_i}")
#         for j in range(p_i):
#             # знаходимо x0
#             if j == 0:
#                 beta_pow_n_div_p_i = pow(base=result, exp=int(n/p_i), mod=modulo)
#                 x_for_each_factor[i].append(r_i_j[i].index(beta_pow_n_div_p_i))
#             else:
#                 pass
#                 # # знаходимо x_i
#                 # power_of_alpha = 0
#                 # for index in range(j):
#                 #     print(f"index {index}")
#                 #     power_of_alpha -= (x_for_each_factor[i][index] * p_i ** index)
#                 #     alpha = pow(base=base, exp=power_of_alpha, mod=modulo)
#                 #     print(f"power of alpha is {power_of_alpha} for x{j}")
#                 #     number = pow(base=int(result*alpha), exp=int(n/p_i**(j+1)), mod=modulo)
#                 #     print(f'number is {number} for x{j}')
#                 #     x_for_each_factor[i].append(r_i_j[i].index(number))
#         print(x_for_each_factor)



def silver_polig_hellman(base, result, modulo):
    factors = [2, 2, 3, 3, 3, 3, 5, 5]
    distinct_factors = sorted(list(set(factors)))
    powers = [factors.count(x) for x in distinct_factors]
    print("Distinct Factors:", distinct_factors)
    print("Powers:", powers)
    r_i_j = [[] for _ in range(len(distinct_factors))]
    # creating r_ij tables for each p_i
    n = modulo - 1
    for i, p_i in enumerate(distinct_factors):
        for j in range(p_i):
            r_j = pow(base, n * j // p_i, modulo)
            r_i_j[i].append(r_j)
    print("r_i_j:", r_i_j)
    x_for_each_factor = [[] for x in range(len(distinct_factors))]
    # проходимо по всіх p_i
    for i, p_i in enumerate(distinct_factors):
        # для кожного p_i шукаємо x_i
        for j in range(powers[i]):
            # знаходимо x0
            if j == 0:
                beta_pow_n_div_p_i = pow(base=result, exp=int(n/p_i), mod=modulo)
                x_for_each_factor[i].append(r_i_j[i].index(beta_pow_n_div_p_i))
            else:
                # знаходимо x_i
                power_of_alpha = sum(-x * (p_i ** k) for k, x in enumerate(x_for_each_factor[i]))
                alpha = pow(base=base, exp=power_of_alpha, mod=modulo)
                number = pow(base=int(result * alpha), exp=int(n / p_i ** (j + 1)), mod=modulo)
                x_for_each_factor[i].append(r_i_j[i].index(number))

    for i, x in enumerate(x_for_each_factor):
        print(f"p{i} = {distinct_factors[i]} x_ith : {x}")

silver_polig_hellman(6,7531,8101)





