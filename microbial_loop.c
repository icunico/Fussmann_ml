#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define N_VARS 10
#define T_MAX 30000.0
#define DT 0.01
#define N_STEPS ((int)(T_MAX / DT))
#define N_TAIL (N_STEPS / 5) // 20% finale della traiettoria

const double r[3] = {3.0, 2.0, 1.0};
const double K[3] = {1.0, 1.0, 1.0};

const double aP1C1 = 9.0, aP1C2 = 3.0, aP2C1 = 2.0, aP2C2 = 6.0, aP3C1 = 1.0, aP3C2 = 3.0;
const double bP1C1 = 5.0, bP1C2 = 5.0, bP2C1 = 5.0, bP2C2 = 5.0, bP3C1 = 5.0, bP3C2 = 5.0;

const double aP1X1 = 0.25, aP2X1 = 0.25, aP3X1 = 0.25, aC1X1 = 3.0, aC2X1 = 3.0;
const double bP1X1 = 0.5,  bP2X1 = 0.5,  bP3X1 = 0.5,  bC1X1 = 2.0, bC2X1 = 2.0;
const double dX1 = 0.2;

const double aX1Y1 = 0.75, bX1Y1 = 0.5, dY1 = 0.1;
const double aY1Z1 = 0.45, bY1Z1 = 0.3, dZ1 = 0.1;
const double tau_val = 50.0;

void derivatives(const double *vars, double *dydt, double dC1, double dC2) {
    double P1 = vars[0], P2 = vars[1], P3 = vars[2];
    double C1 = vars[3], C2 = vars[4];
    double X  = vars[5], Y  = vars[6], Z  = vars[7];
    double D  = vars[8], N  = vars[9];

    double denom_C1 = 1.0 + bP1C1*P1 + bP2C1*P2 + bP3C1*P3;
    double denom_C2 = 1.0 + bP1C2*P1 + bP2C2*P2 + bP3C2*P3;
    double denom_X  = 1.0 + bP1X1*P1 + bP2X1*P2 + bP3X1*P3 + bC1X1*C1 + bC2X1*C2;

    if (denom_C1 < 1e-12) denom_C1 = 1e-12;
    if (denom_C2 < 1e-12) denom_C2 = 1e-12;
    if (denom_X  < 1e-12) denom_X  = 1e-12;

    double denom_N0 = (N + K[0] <= 0) ? 1e-12 : (N + K[0]);
    double denom_N1 = (N + K[1] <= 0) ? 1e-12 : (N + K[1]);
    double denom_N2 = (N + K[2] <= 0) ? 1e-12 : (N + K[2]);

    dydt[0] = P1 * (r[0]*N/denom_N0 - (aP1C1*C1)/denom_C1 - (aP1C2*C2)/denom_C2 - (aP1X1*X)/denom_X);
    dydt[1] = P2 * (r[1]*N/denom_N1 - (aP2C1*C1)/denom_C1 - (aP2C2*C2)/denom_C2 - (aP2X1*X)/denom_X);
    dydt[2] = P3 * (r[2]*N/denom_N2 - (aP3C1*C1)/denom_C1 - (aP3C2*C2)/denom_C2 - (aP3X1*X)/denom_X);

    dydt[3] = C1 * (((aP1C1*P1 + aP2C1*P2 + aP3C1*P3)/denom_C1) - (aC1X1*X)/denom_X - dC1);
    dydt[4] = C2 * (((aP1C2*P1 + aP2C2*P2 + aP3C2*P3)/denom_C2) - (aC2X1*X)/denom_X - dC2);

    dydt[5] = X * (((aP1X1*P1 + aP2X1*P2 + aP3X1*P3 + aC1X1*C1 + aC2X1*C2)/denom_X) - (aX1Y1*Y)/(1.0 + bX1Y1*X) - dX1);
    dydt[6] = Y * (((aX1Y1*X)/(1.0 + bX1Y1*X)) - (aY1Z1*Z)/(1.0 + bY1Z1*Y) - dY1);
    dydt[7] = Z * (((aY1Z1*Y)/(1.0 + bY1Z1*Y)) - dZ1);

    double death_terms = dC1*C1 + dC2*C2 + dX1*X + dY1*Y + dZ1*Z;
    dydt[8] = death_terms - (1.0/tau_val)*D;
    dydt[9] = (1.0/tau_val)*D - N * (P1*r[0]/denom_N0 + P2*r[1]/denom_N1 + P3*r[2]/denom_N2);
}

void rk4_step(double *vars, double dt, double dC1, double dC2) {
    double k1[N_VARS], k2[N_VARS], k3[N_VARS], k4[N_VARS], temp[N_VARS];

    derivatives(vars, k1, dC1, dC2);
    for(int i=0; i<N_VARS; i++) temp[i] = vars[i] + 0.5 * dt * k1[i];
    
    derivatives(temp, k2, dC1, dC2);
    for(int i=0; i<N_VARS; i++) temp[i] = vars[i] + 0.5 * dt * k2[i];

    derivatives(temp, k3, dC1, dC2);
    for(int i=0; i<N_VARS; i++) temp[i] = vars[i] + dt * k3[i];

    derivatives(temp, k4, dC1, dC2);

    for(int i=0; i<N_VARS; i++) {
        vars[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
}

int main() {
    double dc1_min = 0.0, dc1_max = 1.6;
    double dc2_min = 0.0, dc2_max = 1.0;

    int n_dc1 = 100;
    int n_dc2 = 100;

    double *dc1_vals = malloc(n_dc1 * sizeof(double));
    double *dc2_vals = malloc(n_dc2 * sizeof(double));

    if (!dc1_vals || !dc2_vals) {
        printf("Errore nell'allocazione dei parametri.\n");
        return 1;
    }

    for (int i = 0; i < n_dc1; i++) {
        dc1_vals[i] = (n_dc1 > 1) ? (dc1_min + i * (dc1_max - dc1_min) / (n_dc1 - 1)) : dc1_min;
    }

    for (int i = 0; i < n_dc2; i++) {
        dc2_vals[i] = (n_dc2 > 1) ? (dc2_min + i * (dc2_max - dc2_min) / (n_dc2 - 1)) : dc2_min;
    }

    const int specie_ctrl[5] = {0, 1, 2, 3, 4}; 
    const int N_SPECIE_CTRL = 5;

    FILE *f_ss = fopen("valori_equilibrio_steady_state.txt", "w");
    FILE *f_plot = fopen("punti_grafico.txt", "w");

    if (!f_ss || !f_plot) {
        printf("Errore durante l'apertura dei file di output.\n");
        return 1;
    }

    fprintf(f_ss, "dC1\tdC2\tP1\tP2\tP3\tC1\tC2\tX\tY\tZ\tD\tN\n");
    fprintf(f_plot, "dC1\tdC2\tStato\n"); 

    double *history = malloc(N_TAIL * N_VARS * sizeof(double));
    if (!history) {
        printf("Errore nell'allocazione della memoria history.\n");
        return 1;
    }

    int ext_counters[5];

    printf("=== INIZIO SIMULAZIONE E STAMPA DELLE COMBINAZIONI ===\n");

    for (int i1 = 0; i1 < n_dc1; i1++) {
        for (int i2 = 0; i2 < n_dc2; i2++) {
            double dc1 = dc1_vals[i1];
            double dc2 = dc2_vals[i2];

            double vars[N_VARS] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0};
            int step_start_tail = N_STEPS - N_TAIL;

            bool flag_extin = false;
            bool flag_nan = false;

            for (int k = 0; k < N_SPECIE_CTRL; k++) ext_counters[k] = 0;

            // Flag per verificare se questa coppia equivale a (dC1=0.5, dC2=0.2)
            bool save_trajectory = (fabs(dc1 - 0.5) < 0.01) && (fabs(dc2 - 0.2) < 0.01);
            FILE *f_traj = NULL;
            if (save_trajectory) {
                f_traj = fopen("traiettoria_0.5_0.2.txt", "w");
                if (f_traj) {
                    fprintf(f_traj, "t\tP1\tP2\tP3\tC1\tC2\tX\tY\tZ\tD\tN\n");
                }
            }

            // INTEGRAZIONE TEMPORALE
            for (int step = 0; step < N_STEPS; step++) {
                double current_time = step * DT;

                if (f_traj) {
                    fprintf(f_traj, "%.2f", current_time);
                    for (int v = 0; v < N_VARS; v++) {
                        fprintf(f_traj, "\t%.6f", vars[v]);
                    }
                    fprintf(f_traj, "\n");
                }

                rk4_step(vars, DT, dc1, dc2);

                for (int v = 0; v < N_VARS; v++) {
                    if (isnan(vars[v]) || isinf(vars[v])) {
                        flag_nan = true;
                        break;
                    }
                }
                if (flag_nan) break;

                for (int s = 0; s < N_SPECIE_CTRL; s++) {
                    int iv = specie_ctrl[s];
                    if (vars[iv] < 0.001) {
                        ext_counters[s]++;
                        if (ext_counters[s] >= 20000) {
                            flag_extin = true;
                            break; 
                        }
                    } else {
                        ext_counters[s] = 0;
                    }
                }

                if (flag_extin) break;

                if (step >= step_start_tail) {
                    int tail_idx = step - step_start_tail;
                    for (int v = 0; v < N_VARS; v++) {
                        history[tail_idx * N_VARS + v] = vars[v];
                    }
                }
            }

            if (f_traj) {
                fclose(f_traj);
                printf("--> Salvata la traiettoria temporale per dC1=%.2f, dC2=%.2f\n", dc1, dc2);
            }

            // DETERMINAZIONE STATO E STAMPA SU TERMINALE
            int stato = 0;

            if (flag_nan) {
                stato = 3;
            } else if (flag_extin) {
                stato = 0;
            } else {
                bool flag_neg = false;
                for (int s = 0; s < N_SPECIE_CTRL; s++) {
                    int iv = specie_ctrl[s];
                    for (int t = 0; t < N_TAIL; t++) {
                        if (history[t * N_VARS + iv] < -0.01) {
                            flag_neg = true;
                            break;
                        }
                    }
                    if (flag_neg) break;
                }

                if (flag_neg) {
                    stato = 3;
                } else {
                    double max_ratio = 0.0;
                    for (int s = 0; s < N_SPECIE_CTRL; s++) {
                        int iv = specie_ctrl[s];
                        double sum = 0.0;

                        for (int t = 0; t < N_TAIL; t++) {
                            sum += history[t * N_VARS + iv];
                        }
                        double mean = sum / N_TAIL;

                        double sum_sq = 0.0;
                        for (int t = 0; t < N_TAIL; t++) {
                            double diff = history[t * N_VARS + iv] - mean;
                            sum_sq += diff * diff;
                        }
                        double std = sqrt(sum_sq / N_TAIL);

                        double ratio = 0.0;
                        if (mean >= 0.01 && !isnan(mean) && !isnan(std)) {
                            ratio = std / mean;
                        }

                        if (ratio > max_ratio) max_ratio = ratio;
                    }

                    if (max_ratio <= 0.01) {
                        stato = 1;
                        fprintf(f_ss, "%.4f\t%.4f", dc1, dc2);
                        for (int v = 0; v < N_VARS; v++) {
                            fprintf(f_ss, "\t%.6f", history[(N_TAIL - 1) * N_VARS + v]);
                        }
                        fprintf(f_ss, "\n");
                    } else {
                        stato = 2;
                    }
                }
            }

            // Scrittura file grafico e stampa a schermo
            fprintf(f_plot, "%.4f\t%.4f\t%d\n", dc1, dc2, stato);
            printf("dC1: %6.4f | dC2: %6.4f ---> Stato: %d\n", dc1, dc2, stato);
        }
    }

    fclose(f_ss);
    fclose(f_plot);
    free(history);
    free(dc1_vals);
    free(dc2_vals);
    return 0;
}
