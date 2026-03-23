import random
import numpy as np
import networkx as nx


def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        edges = [tuple(map(int, line.strip().split())) for line in lines]
        G = nx.Graph()
        G.add_edges_from(edges)
    return G


def degree_than_one(graph):
    return [node for node, degree in graph.degree() if degree > 1]


def update_parameters(t, max_t, dim):
    a = 2 - 2 * (t / max_t)
    A = np.random.uniform(-a, a, size=(4, dim))
    C = np.random.uniform(0, 2, size=(4, dim))
    return A, C


def adaptive_init(G, V_pie, k, n):
    positions = []
    Sis = []
    _, max_degree = max(G.degree(), key=lambda x: x[1])

    for _ in range(n):
        Xi = []
        for j in V_pie:
            d = G.degree(j)
            if d > 0.5 * max_degree:
                r = random.uniform(0.5 * d, d)
            else:
                r = random.uniform(1, max(d, 1))
            Xi.append(r / max_degree)

        idx = sorted(range(len(Xi)), key=lambda i: Xi[i], reverse=True)[:k]
        Si = [V_pie[i] for i in idx]

        positions.append(Xi)
        Sis.append(Si)

    return np.array(positions), np.array(Sis)


def determine_Si(positions, V_pie, k):
    Sis = []
    for Xi in positions:
        idx = sorted(range(len(Xi)), key=lambda i: Xi[i], reverse=True)[:k]
        Sis.append([V_pie[i] for i in idx])
    return np.array(Sis)


def edv_fitness(S, G, k):
    Ne = set()
    for i in range(k):
        Ne.update(G.neighbors(S[i]))

    Ne -= set(S)
    edv = k

    for u in Ne:
        ti = len(set(G.neighbors(u)) & set(S))
        edv += 1 - (1 - 0.02) ** ti

    return edv


def fitness(Sis, G, k):
    return [edv_fitness(S, G, k) for S in Sis]


def select_best(positions, fitness_values):
    idx = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
    return positions[idx[0]], positions[idx[1]], positions[idx[2]], idx[3:]


def gwo_update(alpha, beta, delta, rand, omega, A, C, w1, w2, w3, w4):
    dim = len(omega)
    new = np.zeros(dim)

    for j in range(dim):
        g1 = alpha[j] - A[0][j] * abs(C[0][j] * alpha[j] - omega[j])
        g2 = beta[j]  - A[1][j] * abs(C[1][j] * beta[j]  - omega[j])
        g3 = delta[j] - A[2][j] * abs(C[2][j] * delta[j] - omega[j])
        g4 = rand[j]  - A[3][j] * abs(C[3][j] * rand[j]  - omega[j])

        new[j] = (w1*g1 + w2*g2 + w3*g3 + w4*g4) / (w1 + w2 + w3 + w4)

    return new


def woa_local(beta, x, A, C):
    for j in range(len(x)):
        x[j] = beta[j] - A[0][j] * abs(C[0][j] * beta[j] - x[j])
    return x


def woa_random(G, V_pie, k, x, A, C):
    temp = adaptive_init(G, V_pie, k, 1)[0][0]
    for j in range(len(x)):
        x[j] = temp[j] - A[0][j] * abs(C[0][j] * temp[j] - x[j])
    return x


def de_update(alpha, beta, delta, rand, k):
    idx_a = sorted(range(len(alpha)), key=lambda i: alpha[i], reverse=True)[:k]
    idx_b = sorted(range(len(beta)), key=lambda i: beta[i], reverse=True)[:k]
    idx_c = sorted(range(len(delta)), key=lambda i: delta[i], reverse=True)[:k]

    inter = set(idx_a) & set(idx_b) & set(idx_c)

    for i in inter:
        rand[i] = alpha[i]

    return rand


def local_search(G, S, k):
    best = S.copy()
    best_fit = edv_fitness(best, G, k)

    avg_deg = sum(dict(G.degree()).values()) / len(G)

    for i in range(len(S)):
        TS = set(G.neighbors(best[i]))
        for n in list(TS):
            TS.update(G.neighbors(n))

        TS -= set(best)
        TS = {x for x in TS if G.degree(x) >= avg_deg}

        for x in TS:
            temp = best.copy()
            temp[i] = x
            f = edv_fitness(temp, G, k)

            if f > best_fit:
                best, best_fit = temp, f

    return best


def HWOAGWO(G, k, n, max_iter, V_pie):
    positions, Sis = adaptive_init(G, V_pie, k, n)
    fit = fitness(Sis, G, k)
    alpha, beta, delta, omega_idx = select_best(positions, fit)

    dim = len(V_pie)

    for t in range(max_iter):
        A, C = update_parameters(t, max_iter, dim)
        p = random.random()

        alpha_S = determine_Si([alpha], V_pie, k)[0]
        beta_S  = determine_Si([beta], V_pie, k)[0]
        delta_S = determine_Si([delta], V_pie, k)[0]

        w1 = edv_fitness(alpha_S, G, k)
        w2 = edv_fitness(beta_S, G, k)
        w3 = edv_fitness(delta_S, G, k)

        for i in range(len(positions)):
            if i in omega_idx:

                rand = positions[random.randint(0, len(positions)-1)]
                rand_S = determine_Si([rand], V_pie, k)[0]
                w4 = edv_fitness(rand_S, G, k)

                if p < 0.5:
                    if abs(A[0][0]) < 1:
                        positions[i] = woa_local(beta, positions[i], A, C)
                    else:
                        temp = adaptive_init(G, V_pie, k, 1)[0][0]
                        positions[i] = de_update(alpha, beta, delta, temp, k)
                else:
                    if abs(A[0][0]) < 1:
                        positions[i] = gwo_update(alpha, beta, delta, rand,
                                                  positions[i], A, C,
                                                  w1, w2, w3, w4)
                    else:
                        positions[i] = woa_random(G, V_pie, k, positions[i], A, C)

        Sis = determine_Si(positions, V_pie, k)
        fit = fitness(Sis, G, k)
        alpha, beta, delta, omega_idx = select_best(positions, fit)

    alpha_S = determine_Si([alpha], V_pie, k)[0]
    return local_search(G, alpha_S, k)


if __name__ == '__main__':
    graph_path = 'CA-HepTh_9877_51971.txt'
    G = read_edge_list(graph_path)
    V_pie = degree_than_one(G)

    for k in [10, 20, 30, 40, 50]:
        res = HWOAGWO(G, k, 50, 80, V_pie)
        print("k =", k, "seed set =", res)
