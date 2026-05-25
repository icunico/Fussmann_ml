from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

def extract_data(file_path):
    """Estrae i dati dal file, gestendo righe vuote e incomplete"""
    data = []

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Salta righe vuote

            # Usa regex per estrarre i dati - ora estraiamo fino alla 7a colonna
            match = re.search(r'configurazione\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+([\d\.]+))?(?:\s+([\d\.]+))?(?:\s+([\d\.]+))?', line)

            if match:
                configurazione = match.group(1)
                legami_onnivori = int(match.group(4))  # 4a colonna

                # Il CAOS è il 7o gruppo (se presente) - colonna 7
                caos_str = match.group(7)  # 7a colonna per CAOS

                if caos_str:
                    try:
                        caos = float(caos_str.replace(',', '.'))

                        # Se è 3B, dobbiamo dividerlo in 3B_1 e 3B_2
                        if configurazione == "3B":
                            data.append({
                                "Configurazione": "3B_1",
                                "Omnivorous_Links": legami_onnivori,
                                "Caos (%)": caos,  # Cambiato da Stabilità a Caos
                                "Linea": line_num,
                                "Originale": "3B"
                            })
                            data.append({
                                "Configurazione": "3B_2",
                                "Omnivorous_Links": legami_onnivori,
                                "Caos (%)": caos,  # Cambiato da Stabilità a Caos
                                "Linea": line_num,
                                "Originale": "3B"
                            })
                        else:
                            data.append({
                                "Configurazione": configurazione,
                                "Omnivorous_Links": legami_onnivori,
                                "Caos (%)": caos,  # Cambiato da Stabilità a Caos
                                "Linea": line_num,
                                "Originale": configurazione
                            })
                    except ValueError:
                        continue

    print(f"\n{'='*60}")
    print(f"ESTRAZIONE COMPLETATA: {len(data)} dati validi estratti")
    print(f"{'='*60}")

    # Raggruppa i dati per configurazione
    gruppi = {}
    for d in data:
        config = d["Configurazione"]
        if config not in gruppi:
            gruppi[config] = []
        gruppi[config].append(d)

    # Ordina ogni gruppo per linea (mantenendo l'ordine originale)
    for config in gruppi:
        gruppi[config].sort(key=lambda x: x["Linea"])

    # DEBUG: mostra quanti dati abbiamo per ogni gruppo
    print("\nDATI PER GRUPPO (CAOS):")
    for config in ["2B", "3B_1", "3B_2", "3C"]:
        if config in gruppi:
            print(f"{config}: {len(gruppi[config])} dati")
            for i, d in enumerate(gruppi[config][:5]):  # Mostra solo primi 5
                print(f"  [{i+1}] Linea {d['Linea']}: Leg.Onn={d['Omnivorous_Links']}, Caos={d['Caos (%)']:.2f}%")
        else:
            print(f"{config}: 0 dati")

    return data, gruppi

def plot_groups_caos(data, gruppi, output_dir):
    """Plot tutti i gruppi richiesti con CAOS vs LEGAMI ONNIVORI"""

    if not data:
        print("ATTENZIONE: Nessun dato da plottare!")
        return

    plt.figure(figsize=(6, 6))

    # DEFINIZIONE DEI GRUPPI - Ora con 3B_1 e 3B_2 separati
    groups_config = [
        {"name": "2A-3A", "groups": ["2A", "3A"], "color": 'gold', "marker": 'o'},
        # Gruppo 1: 2B con 3B_1 - DARKGREEN >
        {"name": "2B-3B-3C", "groups": ["2B", "3B_1"], "color": 'darkgreen', "marker": '>'},
        # Gruppo 2: 3B_2 con 3C - DARKGREEN > (stesso stile)
        {"name": "2B-3B-3C", "groups": ["3B_2", "3C"], "color": 'darkgreen', "marker": '>'},
        {"name": "2D-3G", "groups": ["2D", "3G"], "color": 'deepskyblue', "marker": '^'},
        {"name": "4A-4B-4C", "groups": ["4A", "4B", "4C"], "color": 'magenta', "marker": 'v'},
        {"name": "4D-4E-4F-4G", "groups": ["4D", "4E", "4F", "4G"], "color": 'orange', "marker": 'D'},
        {"name": "5B-5C", "groups": ["5B", "5C"], "color": 'darkred', "marker": '*'}
   ]

    print("\n" + "="*80)
    print("PLOTTING TUTTI I GRUPPI (CAOS):")
    print("="*80)

    points_plotted = 0

    for config_set in groups_config:
        print(f"\nProcessando: {config_set['name']}")
        groups = config_set["groups"]

        # Verifica disponibilità dati
        dati_disponibili = {}
        for g in groups:
            if g in gruppi:
                dati_disponibili[g] = gruppi[g]
                print(f"  {g}: {len(gruppi[g])} dati disponibili")
            else:
                print(f"  {g}: 0 dati disponibili")
                continue

        # Se non abbiamo dati per tutti i gruppi, saltiamo
        if len(dati_disponibili) < 2:
            print(f"  Saltato: dati insufficienti")
            continue  # AGGIUNTO IL CONTINUE MANCANTE QUI!

        # Determina quanti abbinamenti possiamo fare
        counts = [len(dati_disponibili[g]) for g in groups if g in dati_disponibili]
        min_count = min(counts) if counts else 0

        if min_count == 0:
            print(f"  Saltato: nessun abbinamento possibile")
            continue

        print(f"  Abbinamenti possibili: {min_count}")

        # Per ogni configurazione
        for config_idx in range(min_count):
            punti_config = []

            # Raccogli i punti validi per questa configurazione
            for g in groups:
                if g in dati_disponibili and config_idx < len(dati_disponibili[g]):
                    punto = dati_disponibili[g][config_idx]
                    if punto["Caos (%)"] is not None:  # Cambiato da Stabilità a Caos
                        punti_config.append(punto)

            # Se abbiamo almeno 2 punti validi, plottiamo
            if len(punti_config) >= 2:
                # PLOT dei punti
                for punto in punti_config:
                    plt.scatter(punto["Omnivorous_Links"], punto["Caos (%)"],  # Cambiato da Stabilità a Caos
                              color=config_set["color"], marker=config_set["marker"],
                              s=180, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    points_plotted += 1

                # COLLEGAMENTI tra i punti
                punti_ordinati = sorted(punti_config, key=lambda x: x["Omnivorous_Links"])
                x_coords = [p["Omnivorous_Links"] for p in punti_ordinati]
                y_coords = [p["Caos (%)"] for p in punti_ordinati]  # Cambiato da Stabilità a Caos

                linewidth = 3.0 if len(punti_config) <= 2 else 2.5
                plt.plot(x_coords, y_coords, color='black',
                        linestyle='-', linewidth=linewidth, alpha=0.7, zorder=3)

                # Mostra i gruppi collegati
                gruppi_collegati = [p["Configurazione"] for p in punti_config]
                print(f"    Config {config_idx+1}: Collegati {gruppi_collegati}")

    if points_plotted == 0:
        print("ATTENZIONE: Nessun punto plottato!")
        return

    # LEGENDA - Solo una voce per 2B-3B-3C
    from matplotlib.lines import Line2D
    legend_elements = []
    groups_in_legend = set()

    for config in groups_config:
        if config["name"] not in groups_in_legend:
            groups_in_legend.add(config["name"])
            marker_size = 17 if config["name"] == "5B-5C" else 15
            legend_elements.append(
                Line2D([0], [0], marker=config["marker"], color='w',
                       markerfacecolor=config["color"],
                       markersize=marker_size, label=config["name"],
                       markeredgecolor='black', markeredgewidth=1)
            )

    plt.legend(handles=legend_elements, fontsize=16, frameon=True,
               edgecolor='gray', loc='center right', ncol=2,
               fancybox=True, shadow=True)

    # ASSI - CAMBIATE ETICHETTE PER CAOS
    plt.xlabel("Omnivorous Links", fontsize=20, fontweight='bold', labelpad=12)
    plt.ylabel("Chaos (%)", fontsize=20, fontweight='bold', labelpad=12)
    plt.title("Chaos vs Omnivorous Links", fontsize=18, fontweight='bold', pad=15)

    # Formattazione
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Griglia e bordi
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Limiti assi
    plt.ylim(-3, 100)

    # Limiti asse X basati sui dati
    if data:
        min_links = min(d["Omnivorous_Links"] for d in data)
        max_links = max(d["Omnivorous_Links"] for d in data)
        plt.xlim(min_links - 0.5, max_links + 0.5)

        if max_links - min_links <= 10:
            plt.xticks(range(int(min_links), int(max_links) + 1))

    plt.tight_layout()

    output_png = output_dir / 'Chaos_vs_omnivorous_links_all_groups.png'
    output_pdf = output_dir / 'Chaos_vs_omnivorous_links_all_groups.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"\n" + "="*80)
    print(f"RIEPILOGO FINALE:")
    print(f"Punti plottati totali: {points_plotted}")
    print(f"Grafico salvato come:")
    print(f"  - {output_png}")
    print(f"  - {output_pdf}")
    print("="*80)

    plt.close()

# =========================================================
# MAIN
# =========================================================
print("="*80)
print("ANALISI CAOS vs LEGAMI ONNIVORI")  # Cambiato messaggio
print("3B diviso in 3B_1 e 3B_2")
print("="*80)

data, gruppi = extract_data("tabella.txt")
plot_groups_caos(data, gruppi, BASE_DIR)
