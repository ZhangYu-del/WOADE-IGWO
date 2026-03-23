import random
import numpy as np
import networkx as nx
import time


def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        edges = [tuple(map(int, line.strip().split())) for line in lines]
        G = nx.Graph()
        G.add_edges_from(edges)
    return G


def degree_than_one(graph):
    return [node for node, degree in graph.degree() if degree > 1]


def update_parameters(t, max_t):
    a = 2 - 2 * (t / max_t)
    A = np.array([2 * a * random.random() - a for _ in range(3)]).reshape(3, 1)
    C = np.array([2 * random.random() for _ in range(3)]).reshape(3, 1)
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
                r = random.uniform(1, d)
            Xi.append(r / max_degree)
        indices = sorted(range(len(Xi)), key=lambda i: Xi[i], reverse=True)[:k]
        Si = [V_pie[i] for i in indices]
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


def gwo_update(alpha, beta, delta, omega, A, C, w1, w2, w3):
    dim = len(omega)
    new = np.zeros(dim)
    for j in range(dim):
        g1 = alpha[j] - A[0, 0] * abs(C[0, 0] * alpha[j] - omega[j])
        g2 = beta[j] - A[1, 0] * abs(C[1, 0] * beta[j] - omega[j])
        g3 = delta[j] - A[2, 0] * abs(C[2, 0] * delta[j] - omega[j])
        new[j] = (w1 * g1 + w2 * g2 + w3 * g3) / (w1 + w2 + w3)
    return new


def woa_local(beta, x, A, C):
    for j in range(len(x)):
        x[j] = beta[j] - A[0][0] * abs(C[0][0] * beta[j] - x[j])
    return x


def woa_random(G, V_pie, k, x, A, C):
    temp, _ = adaptive_init(G, V_pie, k, 1)
    temp = temp[0]
    for j in range(len(x)):
        x[j] = temp[j] - A[0][0] * abs(C[0][0] * temp[j] - x[j])
    return x


def de_update(alpha, beta, rand, k):
    idx_a = sorted(range(len(alpha)), key=lambda i: alpha[i], reverse=True)[:k]
    idx_b = sorted(range(len(beta)), key=lambda i: beta[i], reverse=True)[:k]
    inter = set(idx_a) & set(idx_b)
    for i in inter:
        rand[i] = alpha[i]
    return rand


def local_search(G, S, T, k):
    best = S.copy()
    best_fit = edv_fitness(best, G, k)
    for i in range(len(S)):
        TS = set(G.neighbors(best[i]))
        for n in list(TS):
            TS.update(G.neighbors(n))
        TS -= set(best)
        TS = {x for x in TS if G.degree(x) >= T}
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

    for t in range(max_iter):
        A, C = update_parameters(t, max_iter)
        p = random.random()

        fit_vals = fitness(determine_Si(positions, V_pie, k), G, k)
        w1, w2, w3 = fit_vals[0], fit_vals[1], fit_vals[2]

        for i in range(len(positions)):
            if i in omega_idx:
                if p < 0.5:
                    if abs(A[0][0]) < 1:
                        positions[i] = woa_local(beta, positions[i], A, C)
                    else:
                        rand = adaptive_init(G, V_pie, k, 1)[0][0]
                        positions[i] = de_update(alpha, beta, rand, k)
                else:
                    if abs(A[0][0]) < 1:
                        positions[i] = gwo_update(alpha, beta, delta, positions[i], A, C, w1, w2, w3)
                    else:
                        positions[i] = woa_random(G, V_pie, k, positions[i], A, C)

        Sis = determine_Si(positions, V_pie, k)
        fit = fitness(Sis, G, k)
        alpha, beta, delta, omega_idx = select_best(positions, fit)

    alpha_S = determine_Si(np.array([alpha]), V_pie, k)[0]
    return local_search(G, alpha_S, max(dict(G.degree()).values()) / 5, k)


if __name__ == '__main__':
    graph = r'CA-HepTh_9877_51971.txt'
    G = read_edge_list(graph)
    V_pie = degree_than_one(G)

    for k in [10, 20, 30, 40, 50]:
        res = HWOAGWO(G, k, 50, 100, V_pie)
        print(k, res)