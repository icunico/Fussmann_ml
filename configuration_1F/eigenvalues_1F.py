import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Parametri Globali del Sistema ---
r = np.array([3.0, 2.0, 1.0])
K = np.array([1.0, 1.0, 1.0])

aP1C1, aP1C2, aP2C1, aP2C2, aP3C1, aP3C2 = 9.0, 3.0, 2.0, 6.0, 1.0, 3.0
bP1C1, bP1C2, bP2C1, bP2C2, bP3C1, bP3C2 = 5.0, 5.0, 5.0, 5.0, 5.0, 5.0

tau_val = 50.0
N_TOT = 6.5


# --- 2. Campo Vettoriale Completo a 7D ---
def calcola_f_7d(v_7d, dc1, dc2):
    p1, p2, p3, c1, c2, d, n = v_7d

    denom_C1 = 1.0 + bP1C1 * p1 + bP2C1 * p2 + bP3C1 * p3
    denom_C2 = 1.0 + bP1C2 * p1 + bP2C2 * p2 + bP3C2 * p3

    # Tassi di assorbimento del nutriente N da parte dei produttori (Michaelis-Menten)
    uptake_P1 = r[0] * n / (n + K[0])
    uptake_P2 = r[1] * n / (n + K[1])
    uptake_P3 = r[2] * n / (n + K[2])

    f = np.zeros(7)

    # Produttori Primari (P1, P2, P3)
    f[0] = p1 * (
        uptake_P1 - (aP1C1 * c1) / denom_C1 - (aP1C2 * c2) / denom_C2
    )
    f[1] = p2 * (
        uptake_P2 - (aP2C1 * c1) / denom_C1 - (aP2C2 * c2) / denom_C2
    )
    f[2] = p3 * (
        uptake_P3 - (aP3C1 * c1) / denom_C1 - (aP3C2 * c2) / denom_C2
    )

    # Consumatori Primari (C1, C2)
    f[3] = c1 * (((aP1C1 * p1 + aP2C1 * p2 + aP3C1 * p3) / denom_C1) - dc1)
    f[4] = c2 * (((aP1C2 * p1 + aP2C2 * p2 + aP3C2 * p3) / denom_C2) - dc2)

    # Detrito (D)
    f[5] = (dc1 * c1 + dc2 * c2) - (1.0 / tau_val) * d

    # Nutriente Azoto (N) esplicito
    f[6] = (1.0 / tau_val) * d - (p1 * uptake_P1 + p2 * uptake_P2 + p3 * uptake_P3)

    return f


def analizza_stabilita_7d(dc1_val, dc2_val, vars_10d, eps=1e-6):
    v_7d_pickle = np.array(
        [
            vars_10d[0],  # P1
            vars_10d[1],  # P2
            vars_10d[2],  # P3
            vars_10d[3],  # C1
            vars_10d[4],  # C2
            vars_10d[8],  # D
            vars_10d[9],  # N
        ]
    )
    v_7d_pickle = np.maximum(0.0, v_7d_pickle)

    f0 = calcola_f_7d(v_7d_pickle, dc1_val, dc2_val)
    residuo = np.linalg.norm(f0)

    # Jacobiana 7x7
    J = np.zeros((7, 7))
    for k in range(7):
        v_plus = np.copy(v_7d_pickle)
        v_plus[k] += eps
        f_plus = calcola_f_7d(v_plus, dc1_val, dc2_val)

        v_minus = np.copy(v_7d_pickle)
        v_minus[k] -= eps
        f_minus = calcola_f_7d(v_minus, dc1_val, dc2_val)

        J[:, k] = (f_plus - f_minus) / (2.0 * eps)

    # Calcolo di tutti i 7 autovalori
    eigvals = np.linalg.eigvals(J)
    max_re_lambda = np.max(np.real(eigvals))

    TOL_ZERO = 1e-6
    is_stabile = bool(max_re_lambda < TOL_ZERO)

    return v_7d_pickle, is_stabile, max_re_lambda, residuo, eigvals


# --- 3. Caricamento File Pickle ---
pkname = "fussmann_1F.pickle"
print(f"Caricamento file '{pkname}'...")

with open(pkname, "rb") as infile:
    new_dict = pickle.load(infile)

x_mat = new_dict.get("X", new_dict.get("x"))
y_mat = new_dict.get("Y", new_dict.get("y"))

if x_mat is None or y_mat is None:
    arrays = [v for v in new_dict.values() if isinstance(v, np.ndarray)]
    x_mat, y_mat = arrays[0], arrays[1]

# --- 4. Selezione Punti Steady State ---
flag_ss = y_mat[:, 2]
ss_indices = np.where(flag_ss == 1)[0]
print(f"Punti in Steady State analizzati (Sistema 7D): {len(ss_indices)}\n")

risultati_totali = []
dC1_list, dC2_list, stabili_list = [], [], []

OFFSET_VAR = 4

for idx_count, i in enumerate(ss_indices):
    dc1_v = x_mat[i, 0]
    dc2_v = x_mat[i, 1]

    vars_10d = y_mat[i, OFFSET_VAR : OFFSET_VAR + 10]
    vars_10d = np.nan_to_num(vars_10d, nan=0.0)

    v_pickle, stabile, max_re, residuo, eigvals = analizza_stabilita_7d(
        dc1_v, dc2_v, vars_10d
    )
    risultati_totali.append((dc1_v, dc2_v, v_pickle, stabile, max_re, residuo, eigvals))

    dC1_list.append(dc1_v)
    dC2_list.append(dc2_v)
    stabili_list.append(stabile)

    if idx_count == 0:
        print("=" * 60)
        print(f"ANALISI DETTAGLIATA SISTEMA 7D - PRIMO PUNTO (i={i})")
        print("=" * 60)
        print(f"Parametri: dC1 = {dc1_v:.6f}, dC2 = {dc2_v:.6f}")
        print(f"Stato (P1, P2, P3, C1, C2, D, N): {np.round(v_pickle, 6)}")
        print(f"Residuo ||f(X)||: {residuo:.12e}")
        print(f"Stato Stabilità: {'STABILE' if stabile else 'INSTABILE'}")
        print("-" * 60)
        print("I 7 AUTOVALORI RILEVATI (Re + Im*j):")
        for idx_ev, ev in enumerate(eigvals, 1):
            print(f"  lambda_{idx_ev}: {ev.real:+.10f} {ev.imag:+.10f}j")
        print("=" * 60 + "\n")

# --- 5. Salvataggio Output TXT Completo ---
header = (
    f"{'dC1':<10} {'dC2':<10} "
    f"{'P1':<10} {'P2':<10} {'P3':<10} {'C1':<10} {'C2':<10} {'D':<10} {'N':<10} "
    f"{'Stabile':<10} {'Residuo':<16} "
    f"{'Eigenvalues 7D (Re + Im j)':<110}\n"
    + "-" * 220
)

with open("risultati_stabilita_equilibri_7d.txt", "w") as f:
    f.write(header + "\n")
    for res in risultati_totali:
        dc1_v, dc2_v, v_pickle, stabile, max_re, residuo, eigvals = res
        stato_str = "SI" if stabile else "NO"
        eig_str = " | ".join([f"{ev.real:+.6f}{ev.imag:+.6f}j" for ev in eigvals])

        f.write(
            f"{dc1_v:<10.4f} {dc2_v:<10.4f} "
            f"{v_pickle[0]:<10.4f} {v_pickle[1]:<10.4f} {v_pickle[2]:<10.4f} "
            f"{v_pickle[3]:<10.4f} {v_pickle[4]:<10.4f} {v_pickle[5]:<10.4f} {v_pickle[6]:<10.4f} "
            f"{stato_str:<10} {residuo:<16.10f} "
            f"{eig_str}\n"
        )

print("[OK] Risultati completi a 7D salvati in 'risultati_stabilita_equilibri_7d.txt'")

# --- 6. Grafico Mappa Autovalori ---
if len(dC1_list) > 0:
    dC1 = np.array(dC1_list)
    dC2 = np.array(dC2_list)
    stabili = np.array(stabili_list)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        dC1[~stabili],
        dC2[~stabili],
        color="red",
        s=30,
        marker="s",
        label=r"$\exists\,\mathrm{Re}(\lambda) > 0$ (Unstable)",
        zorder=2,
    )
    ax.scatter(
        dC1[stabili],
        dC2[stabili],
        color="forestgreen",
        s=30,
        marker="s",
        label=r"All $\mathrm{Re}(\lambda) \leq 0$ (Stable)",
        zorder=3,
    )

    # Limiti degli assi
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(0.0, 1.2)

    # Forzatura tacche esplicite sugli assi (1.6 per X, 1.2 per Y)
    ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])

    ax.set_xlabel(r"$d_{C1}$", fontsize=20)
    ax.set_ylabel(r"$d_{C2}$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_box_aspect(1)
    ax.legend(fontsize=15, loc="lower right")

    plt.tight_layout()

    # Salvataggio garantendo che i margini esterni non taglino il 1.2 in cima
    plt.savefig("mappa_autovalori_7d.png", dpi=300, bbox_inches="tight")
    plt.savefig("mappa_autovalori_7d.pdf", format="pdf", bbox_inches="tight")
    print("[OK] Grafici salvati come 'mappa_autovalori_7d.png' e '.pdf'")