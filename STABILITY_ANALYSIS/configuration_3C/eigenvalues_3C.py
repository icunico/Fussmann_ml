import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Parametri Globali del Sistema (Fedeli allo YAML) ---
r = np.array([3.0, 2.0, 1.0])  # r1=3.0, r2=2.0
K = np.array([1.0, 1.0, 1.0])  # K1=1.0, K2=1.0

# Pascolo di C1 su P1 e P2
aP1C1, bP1C1 = 9.0, 5.0
aP2C1, bP2C1 = 2.0, 5.0

# Predazione di X su P1, P2 e C1
aP1X1, bP1X1 = 0.25, 0.5
aP2X1, bP2X1 = 0.25, 0.5
aC1X1, bC1X1 = 3.0, 2.0

tau_val = 50.0
N_TOT = 5.5  # Conservazione della massa totale (5D)


# --- 2. Campo Vettoriale Ridotto a 5D (N algebrico) ---
def calcola_f_5d(v_5d, dc1_val, dx1_val):
    """
    Stato v_5d: [P1, P2, C1, X, D]
    N = N_TOT - (P1 + P2 + C1 + X + D)
    """
    p1, p2, c1, x, d = v_5d

    # Eliminazione algebrica di N tramite bilancio di massa
    n = max(0.0, N_TOT - (p1 + p2 + c1 + x + d))

    # Denominatori delle risposte funzionali Holling Tipo II
    denom_C1 = 1.0 + bP1C1 * p1 + bP2C1 * p2
    denom_X = 1.0 + bP1X1 * p1 + bP2X1 * p2 + bC1X1 * c1

    # Assorbimento del nutriente N da parte di P1 e P2 (Michaelis-Menten)
    uptake_P1 = r[0] * n / (n + K[0])
    uptake_P2 = r[1] * n / (n + K[1])

    f = np.zeros(5)

    # 1. P1 (Produttore Primario 1)
    f[0] = p1 * (uptake_P1 - (aP1C1 * c1) / denom_C1 - (aP1X1 * x) / denom_X)

    # 2. P2 (Produttore Primario 2)
    f[1] = p2 * (uptake_P2 - (aP2C1 * c1) / denom_C1 - (aP2X1 * x) / denom_X)

    # 3. C1 (Consumatore Primario - pascola su P1 e P2)
    f[2] = c1 * (
        ((aP1C1 * p1 + aP2C1 * p2) / denom_C1)
        - (aC1X1 * x) / denom_X
        - dc1_val
    )

    # 4. X (Predatore X1 - consuma P1, P2 e C1)
    f[3] = x * (
        ((aP1X1 * p1 + aP2X1 * p2 + aC1X1 * c1) / denom_X) - dx1_val
    )

    # 5. Detrito (D)
    f[4] = (dc1_val * c1 + dx1_val * x) - (1.0 / tau_val) * d

    return f


def analizza_stabilita_5d(dc1_val, dx1_val, vars_10d, eps=1e-6):
    # Estrazione delle 5 variabili d'interesse: P1 (0), P2 (1), C1 (4), X (5), D (8)
    v_5d_pickle = np.array(
        [
            vars_10d[0],  # P1
            vars_10d[1],  # P2
            vars_10d[4],  # C1
            vars_10d[5],  # X
            vars_10d[8],  # D
        ]
    )
    v_5d_pickle = np.maximum(0.0, v_5d_pickle)

    f0 = calcola_f_5d(v_5d_pickle, dc1_val, dx1_val)
    residuo = np.linalg.norm(f0)

    # Jacobiana 5x5
    J = np.zeros((5, 5))
    for k in range(5):
        v_plus = np.copy(v_5d_pickle)
        v_plus[k] += eps
        f_plus = calcola_f_5d(v_plus, dc1_val, dx1_val)

        v_minus = np.copy(v_5d_pickle)
        v_minus[k] -= eps
        f_minus = calcola_f_5d(v_minus, dc1_val, dx1_val)

        J[:, k] = (f_plus - f_minus) / (2.0 * eps)

    # Calcolo dei 5 autovalori
    eigvals = np.linalg.eigvals(J)
    max_re_lambda = np.max(np.real(eigvals))

    TOL_ZERO = 1e-6
    is_stabile = bool(max_re_lambda < TOL_ZERO)

    return v_5d_pickle, is_stabile, max_re_lambda, residuo, eigvals


# --- 3. Caricamento File Pickle ---
pkname = "fussmann.pickle"
print(f"Caricamento file '{pkname}'...")

infile = open(pkname, "rb")
new_dict = pickle.load(infile)
infile.close()

x_mat = new_dict.get("X", new_dict.get("x"))
y_mat = new_dict.get("Y", new_dict.get("y"))

if x_mat is None or y_mat is None:
    arrays = [v for v in new_dict.values() if isinstance(v, np.ndarray)]
    x_mat, y_mat = arrays[0], arrays[1]

# --- 4. Selezione Punti Steady State ---
flag_ss = y_mat[:, 2]
ss_indices = np.where(flag_ss == 1)[0]
print(f"Punti in Steady State analizzati (Sistema 5D): {len(ss_indices)}\n")

risultati_totali = []
dC1_list, dX1_list, stabili_list = [], [], []

OFFSET_VAR = 4

for idx_count, i in enumerate(ss_indices):
    # Prima colonna = dC1, Seconda colonna = dX1
    dc1_v = x_mat[i, 0]
    dx1_v = x_mat[i, 1]

    vars_10d = y_mat[i, OFFSET_VAR : OFFSET_VAR + 10]
    vars_10d = np.nan_to_num(vars_10d, nan=0.0)

    v_pickle, stabile, max_re, residuo, eigvals = analizza_stabilita_5d(
        dc1_v, dx1_v, vars_10d
    )

    # Ricostruzione algebrica di N
    n_calc = max(0.0, N_TOT - np.sum(v_pickle))
    v_completo_stampa = np.append(v_pickle, n_calc)

    risultati_totali.append(
        (dc1_v, dx1_v, v_completo_stampa, stabile, max_re, residuo, eigvals)
    )

    dC1_list.append(dc1_v)
    dX1_list.append(dx1_v)
    stabili_list.append(stabile)

    # --- STAMPA DETTAGLIATA PRIMO PUNTO ---
    if idx_count == 0:
        print("=" * 60)
        print(f"ANALISI DETTAGLIATA SISTEMA 5D - PRIMO PUNTO (i={i})")
        print("=" * 60)
        print(f"Parametri: dC1 = {dc1_v:.6f}, dX1 = {dx1_v:.6f}")
        print(
            f"Stato (P1, P2, C1, X, D, N_calc): {np.round(v_completo_stampa, 6)}"
        )
        print(f"Residuo ||f(X)||: {residuo:.12e}")
        print(f"Stato Stabilità: {'STABILE' if stabile else 'INSTABILE'}")
        print("-" * 60)
        print("I 5 AUTOVALORI RILEVATI (Re + Im*j):")
        for idx_ev, ev in enumerate(eigvals, 1):
            print(f"  lambda_{idx_ev}: {ev.real:+.10f} {ev.imag:+.10f}j")
        print("=" * 60 + "\n")

# --- 5. Salvataggio Output TXT Completo ---
header = (
    f"{'dC1':<10} {'dX1':<10} "
    f"{'P1':<10} {'P2':<10} {'C1':<10} {'X':<10} {'D':<10} {'N(calc)':<10} "
    f"{'Stabile':<10} {'Residuo':<16} "
    f"{'Eigenvalues 5D (Re + Im j)':<90}\n"
    + "-" * 185
)

with open("risultati_stabilita_equilibri_5d.txt", "w") as f:
    f.write(header + "\n")
    for res in risultati_totali:
        dc1_v, dx1_v, v_stampa, stabile, max_re, residuo, eigvals = res
        stato_str = "SI" if stabile else "NO"
        eig_str = " | ".join(
            [f"{ev.real:+.6f}{ev.imag:+.6f}j" for ev in eigvals]
        )

        f.write(
            f"{dc1_v:<10.4f} {dx1_v:<10.4f} "
            f"{v_stampa[0]:<10.4f} {v_stampa[1]:<10.4f} {v_stampa[2]:<10.4f} "
            f"{v_stampa[3]:<10.4f} {v_stampa[4]:<10.4f} {v_stampa[5]:<10.4f} "
            f"{stato_str:<10} {residuo:<16.10f} "
            f"{eig_str}\n"
        )

print("[OK] Risultati salvati in 'risultati_stabilita_equilibri_5d.txt'")

# --- 6. Grafico Mappa Autovalori (dC1 vs dX1) ---
if len(dC1_list) > 0:
    dC1 = np.array(dC1_list)
    dX1 = np.array(dX1_list)
    stabili = np.array(stabili_list)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Punti INSTABILI (Rosso)
    ax.scatter(
        dC1[~stabili],
        dX1[~stabili],
        color="red",
        s=30,
        marker="s",
        label=r"$\exists\,\mathrm{Re}(\lambda) > 0$ (Unstable)",
        zorder=2,
    )

    # Punti STABILI (Verde Foresta)
    ax.scatter(
        dC1[stabili],
        dX1[stabili],
        color="forestgreen",
        s=30,
        marker="s",
        label=r"All $\mathrm{Re}(\lambda) \leq 0$ (Stable)",
        zorder=3,
    )

    # Limiti degli assi impostati: dC1 da 0 a 1.6, dX1 da 0 a 0.4
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(0.0, 0.4)

    ax.set_xlabel(r"$d_{C1}$", fontsize=20)
    ax.set_ylabel(r"$d_{X1}$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    # Legenda in basso a destra
    ax.legend(fontsize=15, loc="lower right")

    plt.tight_layout()

    # Salvataggio in PNG e PDF
    plt.savefig("mappa_autovalori_5d.png", dpi=300)
    plt.savefig("mappa_autovalori_5d.pdf", format="pdf", bbox_inches="tight")
    print(
        "[OK] Grafici salvati come 'mappa_autovalori_5d.png' e 'mappa_autovalori_5d.pdf'"
    )