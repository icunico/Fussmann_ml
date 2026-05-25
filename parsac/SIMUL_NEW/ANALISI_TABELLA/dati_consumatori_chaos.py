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
                consumatori = int(match.group(3))  # 3a colonna: CONSUMATORI

                # Il CAOS è il 7o gruppo (se presente) - colonna 7
                caos_str = match.group(7)  # 7a colonna per CAOS

                if caos_str:
                    try:
                        caos = float(caos_str.replace(',', '.'))

                        data.append({
                            "Configurazione": configurazione,
                            "Consumatori": consumatori,
                            "Caos (%)": caos,
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

    # Ordina ogni grupo per linea (mantenendo l'ordine originale)
    for config in gruppi:
        gruppi[config].sort(key=lambda x: x["Linea"])

    return data, gruppi

def plot_groups_caos(data, gruppi, output_dir):
    """Plot 8 gruppi con CAOS vs CONSUMATORI"""

    if not data:
        print("ATTENZIONE: Nessun dato da plottare!")
        return

    plt.figure(figsize=(6, 6))

    # DEFINIZIONE DEGLI OTTO GRUPPI
    groups_config = [
        # Gruppo 2A-2D - VERDE SCURO/TRIANGOLI DESTRA
        {"name": "2A-2D", "groups": ["2A", "2D"], "color": 'darkgreen', "marker": '>'},
        # Gruppo 2B-4A - MAGENTA/DIAMANTI
        {"name": "2B-4A", "groups": ["2B", "4A"], "color": 'magenta', "marker": 'D'},
        # Gruppo 3A-3G - ARANCIONE/STELLE
        {"name": "3A-3G", "groups": ["3A", "3G"], "color": 'orange', "marker": '*'},
        # Gruppo 3B-4B - VIOLA/QUADRATI
        {"name": "3B-4B", "groups": ["3B", "4B"], "color": 'purple', "marker": 's'},
        # Gruppo 3C-4C - MARRONE/CERCHI
        {"name": "3C-4C", "groups": ["3C", "4C"], "color": 'saddlebrown', "marker": 'o'},
        # Gruppo 3D-4E - BLU/TRIANGOLI GIÙ
        {"name": "3D-4E", "groups": ["3D", "4E"], "color": 'deepskyblue', "marker": 'v'},
        # Gruppo 3E-4F - GIALLO/ESAGONI
        {"name": "3E-4F", "groups": ["3E", "4F"], "color": 'gold', "marker": 'h'},
        # Gruppo 3F-4G - ROSSO SCURO/PENTAGONI
        {"name": "3F-4G", "groups": ["3F", "4G"], "color": 'darkred', "marker": 'P'}
   ]

    print("\n" + "="*80)
    print("PLOTTING OTTO GRUPPI (CAOS vs CONSUMATORI):")
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
            continue

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
                    if punto["Caos (%)"] is not None:
                        punti_config.append(punto)

            # Se abbiamo almeno 2 punti validi, plottiamo
            if len(punti_config) >= 2:
                # PLOT dei punti
                for punto in punti_config:
                    # Dimensioni marker diverse per tipo
                    if config_set["marker"] == 'D':  # Diamanti
                        marker_size = 200
                    elif config_set["marker"] == '*':  # Stelle
                        marker_size = 220
                    elif config_set["marker"] == 'P':  # Pentagoni
                        marker_size = 210
                    elif config_set["marker"] == 'h':  # Esagoni
                        marker_size = 190
                    elif config_set["marker"] == 's':  # Quadrati
                        marker_size = 195
                    elif config_set["marker"] == 'o':  # Cerchi
                        marker_size = 185
                    else:  # Triangoli
                        marker_size = 180
                    
                    plt.scatter(punto["Consumatori"], punto["Caos (%)"],
                              color=config_set["color"], marker=config_set["marker"],
                              s=marker_size, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    points_plotted += 1

                # COLLEGAMENTI tra i punti
                punti_ordinati = sorted(punti_config, key=lambda x: x["Consumatori"])
                x_coords = [p["Consumatori"] for p in punti_ordinati]
                y_coords = [p["Caos (%)"] for p in punti_ordinati]

                linewidth = 3.0 if len(punti_config) <= 2 else 2.5
                plt.plot(x_coords, y_coords, color='black',
                        linestyle='-', linewidth=linewidth, alpha=0.7, zorder=3)

                # Mostra i gruppi collegati
                gruppi_collegati = [p["Configurazione"] for p in punti_config]
                print(f"    Config {config_idx+1}: Collegati {gruppi_collegati}")

    if points_plotted == 0:
        print("ATTENZIONE: Nessun punto plottato!")
        return

    # LEGENDA - CENTRO DESTRA
    from matplotlib.lines import Line2D
    legend_elements = []
    
    for config in groups_config:
        # Dimensioni marker nella legenda
        if config["marker"] == 'D':  # Diamanti
            marker_size = 17
        elif config["marker"] == '*':  # Stelle
            marker_size = 19
        elif config["marker"] == 'P':  # Pentagoni
            marker_size = 18
        elif config["marker"] == 'h':  # Esagoni
            marker_size = 17
        elif config["marker"] == 's':  # Quadrati
            marker_size = 16
        elif config["marker"] == 'o':  # Cerchi
            marker_size = 16
        else:  # Triangoli
            marker_size = 15
        
        legend_elements.append(
            Line2D([0], [0], marker=config["marker"], color='w',
                   markerfacecolor=config["color"],
                   markersize=marker_size, label=config["name"],
                   markeredgecolor='black', markeredgewidth=1)
        )

    # LEGENDA A CENTRO DESTRA (ora con 8 gruppi)
    plt.legend(handles=legend_elements, fontsize=16, frameon=True,
               edgecolor='gray', loc='center right', ncol=1,
               fancybox=True, shadow=True)

    # ASSI
    plt.xlabel("Consumers", fontsize=20, fontweight='bold', labelpad=12)
    plt.ylabel("Chaos (%)", fontsize=20, fontweight='bold', labelpad=12)
    plt.title("Chaos vs Consumers", fontsize=18, fontweight='bold', pad=15)

    # Formattazione
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # NO GRIGLIA
    plt.grid(False)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Limiti assi
    plt.ylim(-3, 100)

    # Limiti asse X basati sui dati
    if data:
        # Filtra solo i dati dei gruppi che ci interessano
        filtered_data = [d for d in data if d["Configurazione"] in ["2A", "2D", "2B", "4A", "3A", "3G", "3B", "4B", "3C", "4C", "3D", "4E", "3E", "4F", "3F", "4G"]]
        if filtered_data:
            min_consumers = min(d["Consumatori"] for d in filtered_data)
            max_consumers = max(d["Consumatori"] for d in filtered_data)
            plt.xlim(min_consumers - 0.5, max_consumers + 0.5)
            
            if max_consumers - min_consumers <= 10:
                plt.xticks(range(int(min_consumers), int(max_consumers) + 1))

    plt.tight_layout()

    output_png = output_dir / 'Chaos_vs_consumers_eight_groups.png'
    output_pdf = output_dir / 'Chaos_vs_consumers_eight_groups.pdf'
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
print("ANALISI CAOS vs NUMERO CONSUMATORI")
print("Gruppi: 2A-2D, 2B-4A, 3A-3G, 3B-4B, 3C-4C, 3D-4E, 3E-4F e 3F-4G")
print("File: tabella_c.txt")
print("="*80)

data, gruppi = extract_data("tabella_c.txt")
plot_groups_caos(data, gruppi, BASE_DIR)
