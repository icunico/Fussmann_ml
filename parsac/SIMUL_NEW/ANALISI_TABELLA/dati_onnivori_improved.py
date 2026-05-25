import matplotlib.pyplot as plt
import re
import numpy as np

def extract_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            # Stampa ogni riga per verificare cosa stiamo leggendo
            print(f"Riga letta: {line.strip()}")

            if line.strip() == "":
                continue

            # Modifico la regex per renderla più flessibile
            match = re.match(
                r"configurazione\s*(\S+)\s*(\d+)\s*(\d+)\s*(\d+)\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*$",
                line
            )

            if match:
                # Qui controlliamo ogni gruppo per evitare errori con i valori None e NaN
                configurazione = match.group(1)
                lo = int(match.group(4))
                stabilita = float(match.group(5)) if match.group(5) else None
                chaos = float(match.group(7)) if match.group(7) else None
               # Aggiungiamo il dizionario alla lista dei dati
                data.append({
                    "Configurazione": configurazione,
                    "LO": lo,
                    "Stabilità (%)": stabilita,
                    "CHAOS %": chaos
                })
            else:
                print(f"Non corrisponde: {line.strip()}")

    # Stampa i dati estratti per la verifica
    print("\nDati estratti:")
    for item in data:
        print(item)

    return data

def plot_2A_3A_2B_3B_3C_2D_3G_5B_5C_4A_4B_4C_4D_4E_4F_4G(data):
    plt.figure(figsize=(10, 7))

    # ---------- PARTE 1: 2A–3A RIGA PER RIGA (senza media) ----------
    data_2A = [d for d in data if d["Configurazione"] == "2A"]
    data_3A = [d for d in data if d["Configurazione"] == "3A"]

    n = min(len(data_2A), len(data_3A))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 2A e 3A:")
    for i in range(n):
        print(f"Configurazione: 2A, LO: {data_2A[i]['LO']}, Stabilità (%): {data_2A[i]['Stabilità (%)']}, CHAOS (%): {data_2A[i]['CHAOS %']}")
        print(f"Configurazione: 3A, LO: {data_3A[i]['LO']}, Stabilità (%): {data_3A[i]['Stabilità (%)']}, CHAOS (%): {data_3A[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 2A e 3A, includendo NaN
        x = [data_2A[i]["LO"], data_3A[i]["LO"]]
        y = [
            data_2A[i]["Stabilità (%)"] if data_2A[i]["Stabilità (%)"] is not None else np.nan,
            data_3A[i]["Stabilità (%)"] if data_3A[i]["Stabilità (%)"] is not None else np.nan
        ]

        # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="black",
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="2A–3A" if i == 0 else ""
            )

    # ---------- PARTE 2: 2B–3B–3C (senza media e con valori nulli ignorati) ----------
    data_2B = [d for d in data if d["Configurazione"] == "2B"]
    data_3B = [d for d in data if d["Configurazione"] == "3B"]
    data_3C = [d for d in data if d["Configurazione"] == "3C"]

    n = min(len(data_2B), len(data_3B), len(data_3C))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 2B, 3B e 3C:")
    for i in range(n):
        print(f"Configurazione: 2B, LO: {data_2B[i]['LO']}, Stabilità (%): {data_2B[i]['Stabilità (%)']}, CHAOS (%): {data_2B[i]['CHAOS %']}")
        print(f"Configurazione: 3B, LO: {data_3B[i]['LO']}, Stabilità (%): {data_3B[i]['Stabilità (%)']}, CHAOS (%): {data_3B[i]['CHAOS %']}")
        print(f"Configurazione: 3C, LO: {data_3C[i]['LO']}, Stabilità (%): {data_3C[i]['Stabilità (%)']}, CHAOS (%): {data_3C[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 2B, 3B, 3C, includendo NaN
        x = [data_2B[i]["LO"], data_3B[i]["LO"], data_3C[i]["LO"]]
        y = [
            data_2B[i]["Stabilità (%)"] if data_2B[i]["Stabilità (%)"] is not None else np.nan,
            data_3B[i]["Stabilità (%)"] if data_3B[i]["Stabilità (%)"] is not None else np.nan,
            data_3C[i]["Stabilità (%)"] if data_3C[i]["Stabilità (%)"] is not None else np.nan
        ]

        # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="purple",  # colore per 2B, 3B, 3C
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="2B–3B–3C" if i == 0 else ""
            )

    # ---------- PARTE 3: 2D e 3G ----------
    data_2D = [d for d in data if d["Configurazione"] == "2D"]
    data_3G = [d for d in data if d["Configurazione"] == "3G"]

    n = min(len(data_2D), len(data_3G))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 2D e 3G:")
    for i in range(n):
        print(f"Configurazione: 2D, LO: {data_2D[i]['LO']}, Stabilità (%): {data_2D[i]['Stabilità (%)']}, CHAOS (%): {data_2D[i]['CHAOS %']}")
        print(f"Configurazione: 3G, LO: {data_3G[i]['LO']}, Stabilità (%): {data_3G[i]['Stabilità (%)']}, CHAOS (%): {data_3G[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 2D e 3G
        x = [data_2D[i]["LO"], data_3G[i]["LO"]]
        y = [
            data_2D[i]["Stabilità (%)"] if data_2D[i]["Stabilità (%)"] is not None else np.nan,
            data_3G[i]["Stabilità (%)"] if data_3G[i]["Stabilità (%)"] is not None else np.nan
        ]
       # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="blue",  # colore per 2D e 3G
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="2D–3G" if i == 0 else ""
            )

    # ---------- PARTE 4: 5B e 5C ----------
    data_5B = [d for d in data if d["Configurazione"] == "5B"]
    data_5C = [d for d in data if d["Configurazione"] == "5C"]

    n = min(len(data_5B), len(data_5C))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 5B e 5C:")
    for i in range(n):
        print(f"Configurazione: 5B, LO: {data_5B[i]['LO']}, Stabilità (%): {data_5B[i]['Stabilità (%)']}, CHAOS (%): {data_5B[i]['CHAOS %']}")
        print(f"Configurazione: 5C, LO: {data_5C[i]['LO']}, Stabilità (%): {data_5C[i]['Stabilità (%)']}, CHAOS (%): {data_5C[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 5B e 5C
        x = [data_5B[i]["LO"], data_5C[i]["LO"]]
        y = [
            data_5B[i]["Stabilità (%)"] if data_5B[i]["Stabilità (%)"] is not None else np.nan,
            data_5C[i]["Stabilità (%)"] if data_5C[i]["Stabilità (%)"] is not None else np.nan
        ]

        # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="green",  # colore per 5B e 5C
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="5B–5C" if i == 0 else ""
            )

    # ---------- PARTE 5: 4A, 4B, 4C ----------
    data_4A = [d for d in data if d["Configurazione"] == "4A"]
    data_4B = [d for d in data if d["Configurazione"] == "4B"]
    data_4C = [d for d in data if d["Configurazione"] == "4C"]

    n = min(len(data_4A), len(data_4B), len(data_4C))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 4A, 4B e 4C:")
    for i in range(n):
        print(f"Configurazione: 4A, LO: {data_4A[i]['LO']}, Stabilità (%): {data_4A[i]['Stabilità (%)']}, CHAOS (%): {data_4A[i]['CHAOS %']}")
        print(f"Configurazione: 4B, LO: {data_4B[i]['LO']}, Stabilità (%): {data_4B[i]['Stabilità (%)']}, CHAOS (%): {data_4B[i]['CHAOS %']}")
        print(f"Configurazione: 4C, LO: {data_4C[i]['LO']}, Stabilità (%): {data_4C[i]['Stabilità (%)']}, CHAOS (%): {data_4C[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 4A, 4B, e 4C
        x = [data_4A[i]["LO"], data_4B[i]["LO"], data_4C[i]["LO"]]
        y = [
            data_4A[i]["Stabilità (%)"] if data_4A[i]["Stabilità (%)"] is not None else np.nan,
            data_4B[i]["Stabilità (%)"] if data_4B[i]["Stabilità (%)"] is not None else np.nan,
            data_4C[i]["Stabilità (%)"] if data_4C[i]["Stabilità (%)"] is not None else np.nan
        ]

        # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="red",  # colore per 4A, 4B, 4C
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="4A–4B–4C" if i == 0 else ""
            )
    # ---------- PARTE 6: 4D, 4E, 4F, 4G ----------
    data_4D = [d for d in data if d["Configurazione"] == "4D"]
    data_4E = [d for d in data if d["Configurazione"] == "4E"]
    data_4F = [d for d in data if d["Configurazione"] == "4F"]
    data_4G = [d for d in data if d["Configurazione"] == "4G"]

    n = min(len(data_4D), len(data_4E), len(data_4F), len(data_4G))

    # Stampa dei valori nel terminale
    print("\nValori per i gruppi 4D, 4E, 4F, 4G:")
    for i in range(n):
        print(f"Configurazione: 4D, LO: {data_4D[i]['LO']}, Stabilità (%): {data_4D[i]['Stabilità (%)']}, CHAOS (%): {data_4D[i]['CHAOS %']}")
        print(f"Configurazione: 4E, LO: {data_4E[i]['LO']}, Stabilità (%): {data_4E[i]['Stabilità (%)']}, CHAOS (%): {data_4E[i]['CHAOS %']}")
        print(f"Configurazione: 4F, LO: {data_4F[i]['LO']}, Stabilità (%): {data_4F[i]['Stabilità (%)']}, CHAOS (%): {data_4F[i]['CHAOS %']}")
        print(f"Configurazione: 4G, LO: {data_4G[i]['LO']}, Stabilità (%): {data_4G[i]['Stabilità (%)']}, CHAOS (%): {data_4G[i]['CHAOS %']}")

        # Consideriamo la combinazione di valori tra 4D, 4E, 4F e 4G
        x = [data_4D[i]["LO"], data_4E[i]["LO"], data_4F[i]["LO"], data_4G[i]["LO"]]
        y = [
            data_4D[i]["Stabilità (%)"] if data_4D[i]["Stabilità (%)"] is not None else np.nan,
            data_4E[i]["Stabilità (%)"] if data_4E[i]["Stabilità (%)"] is not None else np.nan,
            data_4F[i]["Stabilità (%)"] if data_4F[i]["Stabilità (%)"] is not None else np.nan,
            data_4G[i]["Stabilità (%)"] if data_4G[i]["Stabilità (%)"] is not None else np.nan
        ]

        # Traccia solo se almeno uno dei valori è valido
        if not all(np.isnan(val) for val in y):  # Se almeno uno è valido, tracciamo
            plt.plot(
                x, y,
                color="orange",  # colore per 4D, 4E, 4F, 4G
                marker="o",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label="4D–4E–4F–4G" if i == 0 else ""
            )

    # ---------- GRAFICA ----------
    plt.xlabel("Omnivorous Links")
    plt.ylabel("Stabilità (%)")
    plt.title("Stabilità vs Omnivorous Links per 2A, 3A, 2B, 3B, 3C, 2D, 3G, 5B, 5C, 4A, 4B, 4C, 4D, 4E, 4F, 4G")
    plt.grid(True)
    plt.legend()
    plt.show()

# =========================================================
# MAIN
# =========================================================
data = extract_data("tabella.txt")
plot_2A_3A_2B_3B_3C_2D_3G_5B_5C_4A_4B_4C_4D_4E_4F_4G(data)
