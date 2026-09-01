import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Parametri Globali del Sistema ---
r = np.array([3.0, 2.0, 1.0])
K = np.array([1.0, 1.0, 1.0])

aP1C1, aP1C2, aP2C1, aP2C2 = 9.0, 3.0, 2.0, 6.0
bP1C1, bP1C2, bP2C1, bP2C2 = 5.0, 5.0, 5.0, 5.0

tau_val = 50.0
N_TOT = 6.0


# --- 2. Campo Vettoriale Ridotto a 5D (m - 1 con P3 = 0) ---
def calcola_f_5d_ridotta(v_5d, dc1, dc2):
    p1, p2, c1, c2, d = v_5d

    # Conservazione della massa: N eliminato algebricamente
    n = N_TOT - (p1 + p2 + c1 + c2 + d)
    n = max(0.0, n)  # Guardrail per consistenza fisica

    denom_C1 = 1.0 + bP1C1 * p1 + bP2C1 * p2
    denom_C2 = 1.0 + bP1C2 * p1 + bP2C2 * p2

    uptake_P1 = r[0] * n / (n + K[0])
    uptake_P2 = r[1] * n / (n + K[1])

    f = np.zeros(5)

    # Produttori Primari (P1, P2)
    f[0] = p1 * (uptake_P1 - (aP1C1 * c1) / denom_C1 - (aP1C2 * c2) / denom_C2)
    f[1] = p2 * (uptake_P2 - (aP2C1 * c1) / denom_C1 - (aP2C2 * c2) / denom_C2)

    # Consumatori Primari (C1, C2)
    f[2] = c1 * (((aP1C1 * p1 + aP2C1 * p2) / denom_C1) - dc1)
    f[3] = c2 * (((aP1C2 * p1 + aP2C2 * p2) / denom_C2) - dc2)

    # Detrito (D)
    f[4] = (dc1 * c1 + dc2 * c2) - (1.0 / tau_val) * d

    return f


def analizza_stabilita_5d_ridotta(dc1_val, dc2_val, vars_10d, eps=1e-6):
    # Vettore delle sole 5 variabili indipendenti [P1, P2, C1, C2, D]
    v_5d_pickle = np.array(
        [
            vars_10d[0],  # P1
            vars_10d[1],  # P2
            vars_10d[3],  # C1
            vars_10d[4],  # C2
            vars_10d[8],  # D
        ]
    )
    v_5d_pickle = np.maximum(0.0, v_5d_pickle)

    f0 = calcola_f_5d_ridotta(v_5d_pickle, dc1_val, dc2_val)
    residuo = np.linalg.norm(f0)

    # Jacobiana 5x5 (m - 1)
    J = np.zeros((5, 5))
    for k in range(5):
        v_plus = np.copy(v_5d_pickle)
        v_plus[k] += eps
        f_plus = calcola_f_5d_ridotta(v_plus, dc1_val, dc2_val)

        v_minus = np.copy(v_5d_pickle)
        v_minus[k] -= eps
        f_minus = calcola_f_5d_ridotta(v_minus, dc1_val, dc2_val)

        J[:, k] = (f_plus - f_minus) / (2.0 * eps)

    # Calcolo dei 5 autovalori indipendenti
    eigvals = np.linalg.eigvals(J)
    max_re_lambda = np.max(np.real(eigvals))

    TOL_ZERO = 1e-6
    is_stabile = bool(max_re_lambda < TOL_ZERO)

    return v_5d_pickle, is_stabile, max_re_lambda, residuo, eigvals


# --- 3. Caricamento File Pickle ---
pkname = "fussmann.pickle"
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
print(f"Punti in Steady State analizzati (Sistema Ridotto 5D m-1): {len(ss_indices)}\n")

risultati_totali = []
dC1_list, dC2_list, stabili_list = [], [], []

OFFSET_VAR = 4

for idx_count, i in enumerate(ss_indices):
    dc1_v = x_mat[i, 0]
    dc2_v = x_mat[i, 1]

    vars_10d = y_mat[i, OFFSET_VAR : OFFSET_VAR + 10]
    vars_10d = np.nan_to_num(vars_10d, nan=0.0)

    v_pickle, stabile, max_re, residuo, eigvals = analizza_stabilita_5d_ridotta(
        dc1_v, dc2_v, vars_10d
    )
    risultati_totali.append((dc1_v, dc2_v, v_pickle, stabile, max_re, residuo, eigvals))

    dC1_list.append(dc1_v)
    dC2_list.append(dc2_v)
    stabili_list.append(stabile)

    if idx_count == 0:
        print("=" * 60)
        print(f"ANALISI DETTAGLIATA SISTEMA RIDOTTO 5D (m-1, P3=0) - PRIMO PUNTO (i={i})")
        print("=" * 60)
        print(f"Parametri: dC1 = {dc1_v:.6f}, dC2 = {dc2_v:.6f}")
        print(f"Stato (P1, P2, C1, C2, D): {np.round(v_pickle, 6)}")
        print(f"Residuo ||f(X)||: {residuo:.12e}")
        print(f"Stato Stabilità: {'STABILE' if stabile else 'INSTABILE'}")
        print("-" * 60)
        print("I 5 AUTOVALORI RILEVATI (Re + Im*j):")
        for idx_ev, ev in enumerate(eigvals, 1):
            print(f"  lambda_{idx_ev}: {ev.real:+.10f} {ev.imag:+.10f}j")
        print("=" * 60 + "\n")

# --- 5. Salvataggio Output TXT Completo ---
header = (
    f"{'dC1':<10} {'dC2':<10} "
    f"{'P1':<10} {'P2':<10} {'C1':<10} {'C2':<10} {'D':<10} "
    f"{'Stabile':<10} {'Residuo':<16} "
    f"{'Eigenvalues 5D Ridotta (Re + Im j)':<80}\n"
    + "-" * 180
)

with open("risultati_stabilita_equilibri_5d_ridotta.txt", "w") as f:
    f.write(header + "\n")
    for res in risultati_totali:
        dc1_v, dc2_v, v_pickle, stabile, max_re, residuo, eigvals = res
        stato_str = "SI" if stabile else "NO"
        eig_str = " | ".join([f"{ev.real:+.6f}{ev.imag:+.6f}j" for ev in eigvals])

        f.write(
            f"{dc1_v:<10.4f} {dc2_v:<10.4f} "
            f"{v_pickle[0]:<10.4f} {v_pickle[1]:<10.4f} "
            f"{v_pickle[2]:<10.4f} {v_pickle[3]:<10.4f} {v_pickle[4]:<10.4f} "
            f"{stato_str:<10} {residuo:<16.10f} "
            f"{eig_str}\n"
        )

print("[OK] Risultati salvati in 'risultati_stabilita_equilibri_5d_ridotta.txt'")

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

    # Limiti rigidi degli assi
    ax.set_xlim(0.0, 1.7)
    ax.set_ylim(0.0, 1.2)

    # Tacche esplicite sugli assi (garantisce visibilità di 1.7 su X e 1.2 su Y)
    ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.7])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])

    ax.set_xlabel(r"$d_{C1}$", fontsize=20)
    ax.set_ylabel(r"$d_{C2}$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_box_aspect(1)
    ax.legend(fontsize=15, loc="lower right")

    plt.tight_layout()

    plt.savefig("mappa_autovalori_5d_ridotta.png", dpi=300, bbox_inches="tight")
    plt.savefig("mappa_autovalori_5d_ridotta.pdf", format="pdf", bbox_inches="tight")
    print("[OK] Grafici salvati correttamente come 'mappa_autovalori_5d_ridotta.png' e '.pdf'")