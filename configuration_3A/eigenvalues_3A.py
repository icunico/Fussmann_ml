import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Parametri Globali del Sistema (Costanti) ---
r = np.array([3.0, 2.0, 1.0])
K = np.array([1.0, 1.0, 1.0])

# Parametri di interazione / Risposte funzionali
aP1C2, bP1C2 = 3.0, 5.0
aP1X1, bP1X1 = 0.25, 0.5

# Legame di ONNIVORIA: X1 consuma C2 (predazione su consumatore)
aC2X1, bC2X1 = 3.0, 2.0

tau_val = 50.0
N_TOT = 5.5  # Conservazione della massa totale


# --- 2. Campo Vettoriale 4D con ONNIVORIA (N algebrico) ---
def calcola_f_4d(v_4d, dc2_val, dx1_val):
    """
    Stato v_4d: [P1, C2, X, D]
    Modello con Onnivoria: X1 predà sia P1 (produttore) che C2 (consumatore).
    """
    p1, c2, x, d = v_4d

    # Bilancio algebrico di massa: N = N_TOT - (P1 + C2 + X + D)
    n = max(0.0, N_TOT - (p1 + c2 + x + d))

    # Risposte funzionali Holling Tipo II con Onnivoria
    denom_C2 = 1.0 + bP1C2 * p1
    # Denominatore per X1: include sia P1 che C2 (ricerca/gestione di entrambe le prede)
    denom_X = 1.0 + bP1X1 * p1 + bC2X1 * c2

    # Assorbimento nutriente Michaelis-Menten da parte di P1
    uptake_P1 = r[0] * n / (n + K[0])

    f = np.zeros(4)

    # 1. P1 (Produttore Primario): Pascolo da C2 + Predazione da X1
    f[0] = p1 * (uptake_P1 - (aP1C2 * c2) / denom_C2 - (aP1X1 * x) / denom_X)

    # 2. C2 (Consumatore Primario): Crescita da P1 - Predazione da X1 (ONNIVORIA) - Morte dc2
    f[1] = c2 * (((aP1C2 * p1) / denom_C2) - (aC2X1 * x) / denom_X - dc2_val)

    # 3. X (Predatore Onnivoro X1): Crescita da P1 e C2 - Morte dx1
    f[2] = x * (((aP1X1 * p1 + aC2X1 * c2) / denom_X) - dx1_val)

    # 4. Detrito (D): Riciclo di mortalità di C2 e X1
    f[3] = (dc2_val * c2 + dx1_val * x) - (1.0 / tau_val) * d

    return f


def analizza_stabilita_4d(dc2_val, dx1_val, vars_10d, eps=1e-6):
    # Estrazione variabili: P1 (0), C2 (4), X (5), D (8)
    v_4d_pickle = np.array([vars_10d[0], vars_10d[4], vars_10d[5], vars_10d[8]])
    v_4d_pickle = np.maximum(0.0, v_4d_pickle)

    f0 = calcola_f_4d(v_4d_pickle, dc2_val, dx1_val)
    residuo = np.linalg.norm(f0)

    # Matrice Jacobiana 4x4 numerica
    J = np.zeros((4, 4))
    for k in range(4):
        v_plus = np.copy(v_4d_pickle)
        v_plus[k] += eps
        f_plus = calcola_f_4d(v_plus, dc2_val, dx1_val)

        v_minus = np.copy(v_4d_pickle)
        v_minus[k] -= eps
        f_minus = calcola_f_4d(v_minus, dc2_val, dx1_val)

        J[:, k] = (f_plus - f_minus) / (2.0 * eps)

    # Autovalori per il criterio di stabilità
    eigvals = np.linalg.eigvals(J)
    max_re_lambda = np.max(np.real(eigvals))

    TOL_ZERO = 1e-6
    is_stabile = bool(max_re_lambda < TOL_ZERO)

    return v_4d_pickle, is_stabile, max_re_lambda, residuo, eigvals


# --- 3. Caricamento File Pickle ---
pkname = "fussmann.pickle"
print(f"Caricamento file '{pkname}'...")

try:
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
    print(
        f"Punti in Steady State analizzati con legame di Onnivoria: {len(ss_indices)}\n"
    )

    risultati_totali = []
    dC2_list, dX1_list, stabili_list = [], [], []

    OFFSET_VAR = 4

    for idx_count, i in enumerate(ss_indices):
        dc2_v = x_mat[i, 0]
        dx1_v = x_mat[i, 1]

        vars_10d = y_mat[i, OFFSET_VAR : OFFSET_VAR + 10]
        vars_10d = np.nan_to_num(vars_10d, nan=0.0)

        v_pickle, stabile, max_re, residuo, eigvals = analizza_stabilita_4d(
            dc2_v, dx1_v, vars_10d
        )

        n_calc = max(0.0, N_TOT - np.sum(v_pickle))
        v_completo_stampa = np.append(v_pickle, n_calc)

        risultati_totali.append(
            (dc2_v, dx1_v, v_completo_stampa, stabile, max_re, residuo, eigvals)
        )

        dC2_list.append(dc2_v)
        dX1_list.append(dx1_v)
        stabili_list.append(stabile)

        if idx_count == 0:
            print("=" * 60)
            print(f"ANALISI DETTAGLIATA SISTEMA 4D (ONNIVORIA) - PRIMO PUNTO (i={i})")
            print("=" * 60)
            print(f"Parametri: dC2 = {dc2_v:.6f}, dX1 = {dx1_v:.6f}")
            print(
                f"Stato (P1, C2, X, D, N_calc): {np.round(v_completo_stampa, 6)}"
            )
            print(f"Residuo ||f(X)||: {residuo:.12e}")
            print(f"Stato Stabilità: {'STABILE' if stabile else 'INSTABILE'}")
            print("-" * 60)
            print("AUTOVALORI RILEVATI:")
            for idx_ev, ev in enumerate(eigvals, 1):
                print(f"  lambda_{idx_ev}: {ev.real:+.10f} {ev.imag:+.10f}j")
            print("=" * 60 + "\n")

    # --- 5. Salvataggio TXT ---
    header = (
        f"{'dC2':<10} {'dX1':<10} "
        f"{'P1':<10} {'C2':<10} {'X':<10} {'D':<10} {'N(calc)':<10} "
        f"{'Stabile':<10} {'Residuo':<16} "
        f"{'Eigenvalues 4D (Re + Im j)':<75}\n"
        + "-" * 165
    )

    with open("risultati_stabilita_equilibri_4d.txt", "w") as f:
        f.write(header + "\n")
        for res in risultati_totali:
            dc2_v, dx1_v, v_stampa, stabile, max_re, residuo, eigvals = res
            stato_str = "SI" if stabile else "NO"
            eig_str = " | ".join(
                [f"{ev.real:+.6f}{ev.imag:+.6f}j" for ev in eigvals]
            )

            f.write(
                f"{dc2_v:<10.4f} {dx1_v:<10.4f} "
                f"{v_stampa[0]:<10.4f} {v_stampa[1]:<10.4f} {v_stampa[2]:<10.4f} "
                f"{v_stampa[3]:<10.4f} {v_stampa[4]:<10.4f} "
                f"{stato_str:<10} {residuo:<16.10f} "
                f"{eig_str}\n"
            )

    print("[OK] Salvato 'risultati_stabilita_equilibri_4d.txt'")

    # --- 6. Grafico Mappa Autovalori ---
    if len(dC2_list) > 0:
        dC2 = np.array(dC2_list)
        dX1 = np.array(dX1_list)
        stabili = np.array(stabili_list)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Punti INSTABILI (Rosso)
        if np.any(~stabili):
            ax.scatter(
                dC2[~stabili],
                dX1[~stabili],
                color="red",
                s=30,
                marker="s",
                label=r"$\exists\,\mathrm{Re}(\lambda) > 0$ (Unstable)",
                zorder=2,
            )

        # Punti STABILI (Verde Foresta)
        if np.any(stabili):
            ax.scatter(
                dC2[stabili],
                dX1[stabili],
                color="forestgreen",
                s=30,
                marker="s",
                label=r"All $\mathrm{Re}(\lambda) \leq 0$ (Stable)",
                zorder=3,
            )

        ax.set_xlabel(r"$d_{C2}$", fontsize=20)
        ax.set_ylabel(r"$d_{X1}$", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_box_aspect(1)
        ax.legend(fontsize=15, loc="lower right")

        plt.tight_layout()
        plt.savefig("mappa_autovalori_4d.png", dpi=300)
        plt.savefig("mappa_autovalori_4d.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

        print("[OK] Mappe salvate in formato PNG e PDF.")

except Exception as e:
    print(f"[ERRORE]: {e}")