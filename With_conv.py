import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv

def encode(bits):
    G1 = [1, 1, 1]
    G2 = [1, 0, 1]
    memory = [0, 0]
    encoded = []
    for bit in bits:
        u = [bit] + memory
        v1 = sum([a * b for a, b in zip(G1, u)]) % 2
        v2 = sum([a * b for a, b in zip(G2, u)]) % 2
        encoded.extend([v1, v2])
        memory = [bit] + memory[:1]
    return encoded

def puncture(bits, pattern):
    result = []
    repeat = len(bits) // len(pattern)
    full_pattern = pattern * repeat
    for b, p in zip(bits, full_pattern):
        if p:
            result.append(b)
    return result

def viterbi_decode_punctured(received_bits, pattern):
    trellis = [{} for _ in range(13)]
    full_bits = []
    j = 0
    for p in pattern:
        if p == 1:
            full_bits.append(received_bits[j])
            j += 1
        else:
            full_bits.append(None)
    trellis[0][0] = {'path': [], 'metric': 0}
    for t in range(12):
        new_trellis = {}
        for state in trellis[t]:
            current_path = trellis[t][state]['path']
            current_metric = trellis[t][state]['metric']
            for input_bit in [0, 1]:
                next_state = ((input_bit << 1) | (state >> 1)) & 3
                out1 = input_bit ^ ((state >> 1) & 1) ^ (state & 1)
                out2 = input_bit ^ (state & 1)
                i1, i2 = 2 * t, 2 * t + 1
                metric = 0
                if full_bits[i1] is not None:
                    metric += int(full_bits[i1] != out1)
                if full_bits[i2] is not None:
                    metric += int(full_bits[i2] != out2)
                total_metric = current_metric + metric
                if next_state not in new_trellis or total_metric < new_trellis[next_state]['metric']:
                    new_trellis[next_state] = {
                        'path': current_path + [input_bit],
                        'metric': total_metric
                    }
        trellis[t + 1] = new_trellis
    final_states = trellis[-1]
    best_state = min(final_states, key=lambda s: final_states[s]['metric'])
    return final_states[best_state]['path']


n_fragments = 8
fragment_bits = 16
experiments = 1000
p_values = np.linspace(0, 0.2, 21)
m_values = [1, 3, 5, 7]

results = {m: [] for m in m_values}
error_stats = {m: {'template': 0, 'id': 0, 'data': 0} for m in m_values}

for m in m_values:
    for p in p_values:
        success_count = 0
        errors_template, errors_id, errors_data = 0, 0, 0

        for _ in range(experiments):
            fragments_ok = 0

            for _ in range(n_fragments):
                detected = False
                for i in range(m):
                    marks = (np.random.rand(12) < p).reshape((4, 3))
                    detected_count = 0
                    for mark in marks:
                        if not np.any(mark):
                            detected_count += 1
                    if detected_count >= 2:
                        detected = True
                        break

                if not detected:
                    errors_template += 1
                    continue

                data_bits = np.ones(fragment_bits - 4, dtype=int)
                encoded_bits = encode(data_bits)
                puncture_pattern = [
                        1, 1, 1, 0,  
                        1, 1, 1, 0,
                        1, 1, 1, 0,
                        1, 1, 1, 0,
                        1, 1, 1, 0, 
                        1, 0, 0, 1  
                    ]
                punctured_bits = puncture(encoded_bits, puncture_pattern)
                received_bits_var=[[] for i in range (0, 3 * m)]
                for i in range (0, 3 * m):
                    corrupted_bits = np.random.rand(len(punctured_bits)) < p
                    result_bits = (np.array(punctured_bits) + corrupted_bits) % 2
                    received_bits_var[i]=result_bits
                received_bits=[]
                for i in range (0, len(received_bits_var[0])):
                    sum_bits = 0
                    for j in range (0, 3 * m):
                        sum_bits += received_bits_var[j][i]
                    bit = 1 if sum_bits > (3 * m) // 2 else 0
                    received_bits.append(bit)
                recovered_bits = viterbi_decode_punctured(received_bits, puncture_pattern)

                ids = []
                for replica_idx in range(3):
                    bits = []
                    for bit_idx in range(3):
                        corrupted = np.random.rand(3 * m) < p
                        votes = 3 * m - np.sum(corrupted)
                        bit = 1 if votes > (3 * m) // 2 else 0
                        bits.append(bit)
                    ids.append(tuple(bits))

                id_counts = Counter(ids)
                id_majority, count = id_counts.most_common(1)[0]
                if count < 2:
                    errors_id += 1
                    continue

                chet = np.random.rand(3 * m) < p
                votes = 3 * m - np.sum(chet)
                chet_majority = 1 if votes > (3 * m) // 2 else 0

                id_value = id_majority[0] * 4 + id_majority[1] * 2 + id_majority[2]
                expected_chet = int(id_value % 2 != 0)
                if chet_majority != expected_chet:
                    errors_id += 1
                    continue

                if all(b == 1 for b in recovered_bits):
                    fragments_ok += 1
                else:
                    errors_data += 1

            if fragments_ok == n_fragments:
                success_count += 1

        P_m = success_count / experiments
        results[m].append(P_m)
        error_stats[m]['template'] += errors_template
        error_stats[m]['id'] += errors_id
        error_stats[m]['data'] += errors_data

output_file = "output.csv"

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["m", "p", "P_m"])

    for m in m_values:
        for i, p in enumerate(p_values):
            writer.writerow([m, round(float(p), 3), round(float(results[m][i]), 6)])


plt.figure(figsize=(10, 6))
for m in m_values:
    plt.plot(p_values, results[m], label=f"m={m}")
plt.xlabel("p")
plt.ylabel("P_m")
plt.grid(True)
plt.legend()
plt.show()

for m in m_values:
    print(f"\nДублирование шаблона (m = {m}):")
    print("  Ошибки в метках шаблона:", error_stats[m]['template'])
    print("  Ошибки в ID и четности:", error_stats[m]['id'])
    print("  Ошибки в информационных битах:", error_stats[m]['data'])
