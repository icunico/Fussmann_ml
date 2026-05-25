from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

def extract_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            match = re.match(
                r"configurazione\s*(\S+)\s*(\d+)\s*(\d+)\s*(\d+)\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?\s*([\d.]+)?",
                line
            )

            if match:
                configurazione = match.group(1)
                trophic_level = int(match.group(2))
                stability = float(match.group(5)) if match.group(5) else None  # Stabilità alla 5a colonna

                data.append({
                    "Configurazione": configurazione,
                    "Trophic_Level": trophic_level,
                    "Stabilità (%)": stability
                })

    return data

def plot_four_groups_stability(data, output_dir):
    """Plot 4 set di gruppi con STABILITÀ invece di caos"""
    plt.figure(figsize=(6, 6))  # Leggermente più grande

    # SET 1: Gruppi A originali
    set1_groups = ["1A", "2A", "5A", "5D"]
    set1_color = 'gold'          # Giallo
    set1_marker = 'o'            # Cerchio

    # SET 2: Gruppi B
    set2_groups = ["1B", "2B"]
    set2_color = 'limegreen'     # Verde
    set2_marker = 's'            # Quadrato

    # SET 3: Gruppo 4A e 1E
    set3_groups = ["4A", "1E"]
    set3_color = 'deepskyblue'   # Azzurro
    set3_marker = '^'            # Triangolo

    # SET 4: Gruppo 1F, 4D, 5B
    set4_groups = ["1F", "4D", "5B"]
    set4_color = 'magenta'       # Magenta
    set4_marker = 'v'            # Triangolo rovesciato

    # Tutti i gruppi unici per estrazione dati
    all_unique_groups = list(set(set1_groups + set2_groups + set3_groups + set4_groups))
    group_data = {}

    print("\n" + "="*80)
    print("ORGANIZZAZIONE DATI STABILITÀ PER TUTTI I GRUPPI:")
    print("="*80)

    for group in all_unique_groups:
        group_datas = [d for d in data if d["Configurazione"] == group]
        group_datas.sort(key=lambda x: x["Trophic_Level"])
        group_data[group] = group_datas

        if group_datas:  # Stampa solo se ci sono dati
            print(f"\n{group} ({len(group_datas)} configurazioni):")
            for i, d in enumerate(group_datas):
                stability_str = f"{d['Stabilità (%)']:.1f}%" if d["Stabilità (%)"] is not None else "NaN"
                print(f"  Configurazione {i+1}: T={d['Trophic_Level']}, Stabilità={stability_str}")

    # Trova numeri minimi per ogni set
    min_configs_set1 = min(len(group_data.get(g, [])) for g in set1_groups if g in group_data)
    min_configs_set2 = min(len(group_data.get(g, [])) for g in set2_groups if g in group_data)
    min_configs_set3 = min(len(group_data.get(g, [])) for g in set3_groups if g in group_data)
    min_configs_set4 = min(len(group_data.get(g, [])) for g in set4_groups if g in group_data)

    print(f"\nNumero MINIMO configurazioni:")
    print(f"  Set 1 ({', '.join(set1_groups)}): {min_configs_set1}")
    print(f"  Set 2 ({', '.join(set2_groups)}): {min_configs_set2}")
    print(f"  Set 3 ({', '.join(set3_groups)}): {min_configs_set3}")
    print(f"  Set 4 ({', '.join(set4_groups)}): {min_configs_set4}")

    # PLOT PUNTI SET 1 (gialli, cerchi) - STABILITÀ
    print("\n" + "="*80)
    print("PLOTTING SET 1 - Gruppi A (gialli, cerchi) - STABILITÀ:")
    print("="*80)

    for config_idx in range(min_configs_set1):
        for group in set1_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    plt.scatter(d["Trophic_Level"], d["Stabilità (%)"],
                              color=set1_color, marker=set1_marker,
                              s=180, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    print(f"  Set1: {group}[{config_idx+1}] -> T={d['Trophic_Level']}, S={d['Stabilità (%)']:.1f}%")

    # PLOT PUNTI SET 2 (verdi, quadrati) - STABILITÀ
    print("\n" + "="*80)
    print("PLOTTING SET 2 - Gruppi B (verdi, quadrati) - STABILITÀ:")
    print("="*80)
    
    for config_idx in range(min_configs_set2):
        for group in set2_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    plt.scatter(d["Trophic_Level"], d["Stabilità (%)"],
                              color=set2_color, marker=set2_marker,
                              s=180, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    print(f"  Set2: {group}[{config_idx+1}] -> T={d['Trophic_Level']}, S={d['Stabilità (%)']:.1f}%")

    # PLOT PUNTI SET 3 (azzurri, triangoli) - STABILITÀ
    print("\n" + "="*80)
    print("PLOTTING SET 3 - Gruppo 4A e 1E (azzurri, triangoli) - STABILITÀ:")
    print("="*80)

    for config_idx in range(min_configs_set3):
        for group in set3_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    plt.scatter(d["Trophic_Level"], d["Stabilità (%)"],
                              color=set3_color, marker=set3_marker,
                              s=180, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    print(f"  Set3: {group}[{config_idx+1}] -> T={d['Trophic_Level']}, S={d['Stabilità (%)']:.1f}%")

    # PLOT PUNTI SET 4 (magenta, triangoli rovesciati) - STABILITÀ
    print("\n" + "="*80)
    print("PLOTTING SET 4 - Gruppo 1F, 4D, 5B (magenta, triangoli rovesciati) - STABILITÀ:")
    print("="*80)

    for config_idx in range(min_configs_set4):
        for group in set4_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    plt.scatter(d["Trophic_Level"], d["Stabilità (%)"],
                              color=set4_color, marker=set4_marker,
                              s=180, edgecolors='black', linewidth=2,
                              alpha=0.9, zorder=5)
                    print(f"  Set4: {group}[{config_idx+1}] -> T={d['Trophic_Level']}, S={d['Stabilità (%)']:.1f}%")

    # COLLEGAMENTI SET 1 - STABILITÀ
    print("\n" + "="*80)
    print("COLLEGAMENTI SET 1 - Gruppi A (STABILITÀ):")
    print("="*80)
    connection_count_set1 = 0

    for config_idx in range(min_configs_set1):
        print(f"\n--- Configurazione Set1-{config_idx+1} ---")
        x_coords = []
        y_coords = []
        valid_groups = []

        for group in set1_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    x_coords.append(d["Trophic_Level"])
                    y_coords.append(d["Stabilità (%)"])
                    valid_groups.append(group)

        if len(x_coords) >= 2:
            sorted_indices = np.argsort(x_coords)
            x_sorted = [x_coords[i] for i in sorted_indices]
            y_sorted = [y_coords[i] for i in sorted_indices]

            plt.plot(x_sorted, y_sorted, color='black',
                    linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            connection_count_set1 += 1
            print(f"  ✓ COLLEGATO: {len(valid_groups)} punti validi")
            print(f"    Gruppi: {', '.join(valid_groups)}")
        else:
            print(f"  ✗ NON COLLEGATO: solo {len(valid_groups)} punto/i valido/i")

    # COLLEGAMENTI SET 2 - STABILITÀ
    print("\n" + "="*80)
    print("COLLEGAMENTI SET 2 - Gruppi B (STABILITÀ):")
    print("="*80)
    connection_count_set2 = 0

    for config_idx in range(min_configs_set2):
        print(f"\n--- Configurazione Set2-{config_idx+1} ---")

        x_coords = []
        y_coords = []
        valid_groups = []

        for group in set2_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    x_coords.append(d["Trophic_Level"])
                    y_coords.append(d["Stabilità (%)"])
                    valid_groups.append(group)

        if len(x_coords) >= 2:
            sorted_indices = np.argsort(x_coords)
            x_sorted = [x_coords[i] for i in sorted_indices]
            y_sorted = [y_coords[i] for i in sorted_indices]
            plt.plot(x_sorted, y_sorted, color='black',
                    linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            connection_count_set2 += 1
            print(f"  ✓ COLLEGATO: {len(valid_groups)} punti validi")
            print(f"    Gruppi: {', '.join(valid_groups)}")
        else:
            print(f"  ✗ NON COLLEGATO: solo {len(valid_groups)} punto/i valido/i")

    # COLLEGAMENTI SET 3 - STABILITÀ
    print("\n" + "="*80)
    print("COLLEGAMENTI SET 3 - Gruppo 4A e 1E (STABILITÀ):")
    print("="*80)
    connection_count_set3 = 0

    for config_idx in range(min_configs_set3):
        print(f"\n--- Configurazione Set3-{config_idx+1} ---")

        x_coords = []
        y_coords = []
        valid_groups = []

        for group in set3_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    x_coords.append(d["Trophic_Level"])
                    y_coords.append(d["Stabilità (%)"])
                    valid_groups.append(group)

        if len(x_coords) >= 2:
            sorted_indices = np.argsort(x_coords)
            x_sorted = [x_coords[i] for i in sorted_indices]
            y_sorted = [y_coords[i] for i in sorted_indices]

            plt.plot(x_sorted, y_sorted, color='black',
                    linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            connection_count_set3 += 1
            print(f"  ✓ COLLEGATO: {len(valid_groups)} punti validi")
            print(f"    Gruppi: {', '.join(valid_groups)}")
        else:
            print(f"  ✗ NON COLLEGATO: solo {len(valid_groups)} punto/i valido/i")

    # COLLEGAMENTI SET 4 - STABILITÀ
    print("\n" + "="*80)
    print("COLLEGAMENTI SET 4 - Gruppo 1F, 4D, 5B (STABILITÀ):")
    print("="*80)
    connection_count_set4 = 0

    for config_idx in range(min_configs_set4):
        print(f"\n--- Configurazione Set4-{config_idx+1} ---")

        x_coords = []
        y_coords = []
        valid_groups = []

        for group in set4_groups:
            if group in group_data and config_idx < len(group_data[group]):
                d = group_data[group][config_idx]
                if d["Stabilità (%)"] is not None:
                    x_coords.append(d["Trophic_Level"])
                    y_coords.append(d["Stabilità (%)"])
                    valid_groups.append(group)

        if len(x_coords) >= 2:
            sorted_indices = np.argsort(x_coords)
            x_sorted = [x_coords[i] for i in sorted_indices]
            y_sorted = [y_coords[i] for i in sorted_indices]

            plt.plot(x_sorted, y_sorted, color='black',
                    linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)

            connection_count_set4 += 1
            print(f"  ✓ COLLEGATO: {len(valid_groups)} punti validi")
            print(f"    Gruppi: {', '.join(valid_groups)}")
        else:
            print(f"  ✗ NON COLLEGATO: solo {len(valid_groups)} punto/i valido/i")

        # LEGENDA
        from matplotlib.lines import Line2D
        legend_elements = [
         Line2D([0], [0], marker='o', color='w', markerfacecolor=set1_color,
             markersize=15, label='1A-2A-5A-5D',
             markeredgecolor='black', markeredgewidth=1),
         Line2D([0], [0], marker='s', color='w', markerfacecolor=set2_color,
             markersize=15, label='1B-2B',
             markeredgecolor='black', markeredgewidth=1),
         Line2D([0], [0], marker='^', color='w', markerfacecolor=set3_color,
             markersize=15, label='4A-1E',
             markeredgecolor='black', markeredgewidth=1),
         Line2D([0], [0], marker='v', color='w', markerfacecolor=set4_color,
             markersize=15, label='1F-4D-5B',
             markeredgecolor='black', markeredgewidth=1),
        ]

        plt.legend(handles=legend_elements, fontsize=16, frameon=True,
             edgecolor='gray', loc='upper right', ncol=2)

        # ASSI PIÙ GRANDI
        plt.xlabel("Trophic Levels", fontsize=20, fontweight='bold', labelpad=12)
        plt.ylabel("Steady State (%)", fontsize=20, fontweight='bold', labelpad=12)
    
    # Titolo
    plt.title("Steady State vs Trophic Levels",
              fontsize=18, fontweight='bold', pad=15)

    # Numeri degli assi più grandi
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Spessori assi
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Aggiusta layout
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.1)

    plt.ylim(-3,100)
    plt.tight_layout()

    output_png = output_dir / 'Stability_vs_trophic_levels.png'
    output_pdf = output_dir / 'Stability_vs_trophic_levels.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"\nGrafico salvato come:")
    print(f"  - {output_png}")
    print(f"  - {output_pdf}")
    plt.close()

# =========================================================
# MAIN
# =========================================================
print("="*80)
print("ANALISI STABILITÀ STEADY STATE PER QUATTRO SET DI GRUPPI")
print("="*80)
data = extract_data(BASE_DIR / "tabella_lt.txt")

plot_four_groups_stability(data, BASE_DIR)
