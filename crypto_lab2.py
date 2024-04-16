import time
from sympy import factorint



def bruteforce_log(base, result, mod):
    # Шукаємо дискретний логарифм
    start_time = time.time()
    for x in range(1, mod):
        if pow(base, x, mod) == result:
            return x, time.time()-start_time
        # Перевіряємо час виконання
        if time.time() - start_time >= 300:  # 5 хвилин
            return None, None
    else:
        return None, None


def chinese_remainder(a,n):
    sum_ = 0
    prod = 1
    for n_i in n:
        prod *= n_i

    for n_i, a_i in zip(n, a):
        p = prod // n_i
        sum_ += a_i * pow(base=p, exp=-1, mod=n_i) * p
    return sum_ % prod


def silver_polig_hellman(base, result, modulo):
    n = modulo - 1
    factor_dict = factorint(n)
    factors = []
    for factor, power in factor_dict.items():
        factors.extend([factor] * power)
    distinct_factors = sorted(list(set(factors)))
    powers = [factors.count(x) for x in distinct_factors]
    modules = [pow(f, p) for f, p in zip(distinct_factors, powers)]
    r_i_j = [[] for _ in range(len(distinct_factors))]
    for i, p_i in enumerate(distinct_factors):
        for j in range(p_i):
            r_j = pow(base, n * j // p_i, modulo)
            r_i_j[i].append(r_j)
    x_for_each_factor = [[] for x in range(len(distinct_factors))]
    for i, p_i in enumerate(distinct_factors):
        for j in range(powers[i]):
            # Знаходимо x0
            if j == 0:
                beta_pow_n_div_p_i = pow(base=result, exp=int(n / p_i), mod=modulo)
                indexes = [index for index, num in enumerate(r_i_j[i]) if num == beta_pow_n_div_p_i]
                x_for_each_factor[i].append(indexes[-1])
            else:
                # Знаходимо x_i
                power_of_alpha = sum(-x * (p_i ** k) for k, x in enumerate(x_for_each_factor[i]))
                alpha = pow(base=base, exp=power_of_alpha, mod=modulo)
                number = pow(base=int(result * alpha), exp=int(n / p_i ** (j + 1)), mod=modulo)
                x_for_each_factor[i].append(r_i_j[i].index(number))
    result_list = []
    for x, p in zip(x_for_each_factor, distinct_factors):
        res = sum(x[i] * (p ** i) for i in range(len(x)))
        result_list.append(res)
    answer = chinese_remainder(a=result_list, n=modules)
    print(f"Answer to equation {base}^x = {result} mod {modulo} is {answer}")


start = time.time()
silver_polig_hellman(1312, 973323, 19949093)
print(f"SPH time : {time.time()-start}")
rs, t = bruteforce_log(base=1312, result=973323, mod=19949093)
print(f"Brute time : {t}")