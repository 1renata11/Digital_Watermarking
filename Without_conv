import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv

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

        for i in range(experiments):
            fragments_ok = 0

            for i in range(n_fragments):
                bit_votes = [0 for j in range(fragment_bits)]

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

                for bit_id in range(fragment_bits):
                    corrupted = np.random.rand(4 * m) < p
                    correct_votes = 4 * m - np.sum(corrupted)
                    majority = 1 if correct_votes > 2 * m else 0
                    bit_votes[bit_id]=majority

                ids = []
                for replica_idx in range(4):
                    bits = []
                    for bit_idx in range(3):
                        corrupted = np.random.rand(4 * m) < p
                        votes = 4 * m - np.sum(corrupted)
                        bit = 1 if votes > 2*m else 0
                        bits.append(bit)
                    ids.append(tuple(bits))

                id_counts = Counter(ids)
                id_majority, count = id_counts.most_common(1)[0]
                if count < 3:
                    errors_id += 1
                    continue

                chet = np.random.rand(4 * m) < p
                votes = 4 * m - np.sum(chet)
                chet_majority = 1 if votes > 2*m else 0

                id_value = id_majority[0] * 4 + id_majority[1] * 2 + id_majority[2]
                expected_chet = int(id_value % 2 != 0)
                if chet_majority != expected_chet:
                    errors_id += 1
                    continue

                if all(b == 1 for b in bit_votes[4:]):
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
