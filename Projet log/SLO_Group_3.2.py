"""
Centralized assignment of EVs to existing charging stations on Swiss highways.

Goal:
- Assign EVs to existing charging stations.
- Find the critical number of cars N for which the system becomes saturated.

Inputs are read from:
- Inputs/

Outputs are saved in:
- Outputs/
"""

# ============================================================
# Imports
# ============================================================

import os
import re
import glob
import sys
import random
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# Folders
# ============================================================

INPUT_DIR = "Inputs"
OUTPUT_DIR = "Outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set to True only if you want to display large tables in the terminal
DISPLAY_TABLES = False


# ============================================================
# Reproducibility
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# Basic environment check
# ============================================================

print("Python executable:", sys.executable)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)


# ============================================================
# Helper functions
# ============================================================

def normalize_filename(filename):
    """
    Normalize a filename to make robust matching possible.
    """
    return re.sub(r"[^a-z0-9]", "", filename.lower())


def extract_km(value):
    """
    Extract the first numeric value from a localization string.
    Example: 'PK 12 km' -> 12.0
    """
    if pd.isna(value):
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def normalize_rest_area_name(s):
    """
    Normalize rest area names for matching between CSV sources.
    """
    if pd.isna(s):
        return ""

    s = str(s).strip().lower()

    replacements = {
        "ouest": "west",
        "est": "ost",
        "sud": "sued",
        "süd": "sued",
        "nord": "nord",
        "ovest": "west",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"[^a-z0-9]", "", s)

    return s


# ============================================================
# Step 1 — Read highway rest-area CSV files from Inputs/
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 — Lecture des CSV autoroutes depuis Inputs/")
print("=" * 60)

all_csv = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
print("CSV trouvés :", all_csv)

expected_map = {
    ("A2", +1): "a2balecomocsv",
    ("A2", -1): "a2comobalecsv",
    ("A8", +1): "a8luzernthuncsv",
    ("A8", -1): "a8thunluzerncsv",
    ("A13", +1): "a13sanktmargrethenbellinzonacsv",
    ("A13", -1): "a13bellinzonasanktmargrethencsv",
}

csv_files = {}

for key, expected_norm in expected_map.items():
    matches = [
        f for f in all_csv
        if normalize_filename(os.path.basename(f)) == expected_norm
    ]

    if len(matches) == 1:
        csv_files[key] = matches[0]
    elif len(matches) == 0:
        print(f"Fichier introuvable pour {key}, attendu = {expected_norm}")
    else:
        print(f"Plusieurs fichiers possibles pour {key} :", matches)

print("\nFichiers retenus :")
for k, v in csv_files.items():
    print(k, "->", v)

if len(csv_files) != 6:
    raise ValueError(
        "Tous les fichiers nécessaires n'ont pas été trouvés automatiquement. "
        "Vérifie les noms des CSV dans le dossier Inputs."
    )

all_rows = []

for (autoroute, sens), path in csv_files.items():
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )

    df.columns = df.columns.str.strip()

    print(f"\nLecture de {path}")
    print("Colonnes détectées :", list(df.columns))

    if "Aire" not in df.columns or "Localisation" not in df.columns:
        raise ValueError(
            f"Le fichier {path} doit contenir les colonnes 'Aire' et 'Localisation'."
        )

    df = df[df["Aire"].notna()].copy()
    df["pk"] = df["Localisation"].apply(extract_km)
    df = df[df["pk"].notna()].copy()

    df["autoroute"] = autoroute
    df["nom"] = df["Aire"].astype(str).str.strip()
    df["sens"] = sens
    df["has_existing_charger"] = False
    df["existing_chargers"] = 0

    df = df[
        [
            "autoroute",
            "nom",
            "pk",
            "sens",
            "has_existing_charger",
            "existing_chargers",
        ]
    ]

    all_rows.append(df)

df_aires = pd.concat(all_rows, ignore_index=True)
df_aires = df_aires.sort_values(["autoroute", "sens", "pk"]).reset_index(drop=True)
df_aires["id"] = range(len(df_aires))

print("\nAperçu final des aires :")
print(df_aires.head().to_string(index=False))
print(f"Nombre total d'aires : {len(df_aires)}")


# ============================================================
# Step 2 — Enrich with existing EV charger data from Inputs/
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 — Enrichissement avec Inputs/EV_chargeurs.csv")
print("=" * 60)

ev_file = os.path.join(INPUT_DIR, "EV_chargeurs.csv")

if not os.path.exists(ev_file):
    raise FileNotFoundError(f"Le fichier {ev_file} est introuvable.")

ev_raw = pd.read_csv(
    ev_file,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

ev_raw = ev_raw.drop(
    columns=[col for col in ev_raw.columns if "Unnamed" in col],
    errors="ignore",
)

ev_raw.columns = ev_raw.columns.str.strip().str.lstrip("\ufeff")

print("Colonnes détectées dans EV_chargeurs.csv :", list(ev_raw.columns))

required_ev_cols = [
    "Highway",
    "sens",
    "Rest stop name",
    "Nb EV charging points",
    "Data status",
]

for col in required_ev_cols:
    if col not in ev_raw.columns:
        raise ValueError(
            f"La colonne '{col}' est absente de {ev_file}. "
            f"Colonnes trouvées : {list(ev_raw.columns)}"
        )

# Statuses to exclude
EXCLUDED_STATUSES = {"Confirmed no charger", "Hors Suisse"}

ev_raw["_exclude"] = ev_raw["Data status"].isin(EXCLUDED_STATUSES)
ev_active = ev_raw[~ev_raw["_exclude"]].copy()

ev_active["highway_norm"] = ev_active["Highway"].astype(str).str.strip().str.upper()
ev_active["sens_norm"] = ev_active["sens"].astype(int)
ev_active["name_norm"] = ev_active["Rest stop name"].apply(normalize_rest_area_name)

ev_active["nb_chargers"] = (
    pd.to_numeric(ev_active["Nb EV charging points"], errors="coerce")
    .fillna(0)
    .astype(int)
)

print(f"\nStations EV actives retenues après exclusion : {len(ev_active)}")

if DISPLAY_TABLES:
    print(
        ev_active[
            [
                "Highway",
                "sens",
                "Rest stop name",
                "nb_chargers",
                "Data status",
            ]
        ].to_string(index=False)
    )

df_aires["highway_norm"] = df_aires["autoroute"].astype(str).str.strip().str.upper()
df_aires["sens_norm"] = df_aires["sens"].astype(int)
df_aires["name_norm"] = df_aires["nom"].apply(normalize_rest_area_name)

df_aires = df_aires.merge(
    ev_active[
        [
            "highway_norm",
            "sens_norm",
            "name_norm",
            "nb_chargers",
            "Data status",
        ]
    ],
    on=["highway_norm", "sens_norm", "name_norm"],
    how="left",
)

df_aires["has_existing_charger"] = (
    df_aires["Data status"].notna()
    & ~df_aires["Data status"].isin(EXCLUDED_STATUSES)
)

df_aires["existing_chargers"] = df_aires["nb_chargers"].fillna(0).astype(int)
df_aires["ev_providers"] = ""
df_aires["known_chargers_comments"] = df_aires["existing_chargers"]

df_aires = df_aires.drop(
    columns=["highway_norm", "sens_norm", "name_norm", "nb_chargers"],
    errors="ignore",
)

df_aires["ev_charger_presence"] = df_aires["has_existing_charger"].map(
    {True: "Oui", False: "Non"}
)

df_aires["existing_chargers_known"] = df_aires["known_chargers_comments"]

df_aires.loc[
    df_aires["ev_charger_presence"] == "Non",
    "existing_chargers_known",
] = 0

df_aires.loc[
    (df_aires["ev_charger_presence"] == "Oui")
    & (df_aires["existing_chargers_known"] == 0),
    "existing_chargers_known",
] = pd.NA

print("\nAperçu enrichi, 30 premières lignes :")
print(
    df_aires[
        [
            "autoroute",
            "nom",
            "sens",
            "has_existing_charger",
            "existing_chargers",
            "ev_charger_presence",
            "Data status",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# Step 3 — Global scenario parameters
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 — Paramètres globaux")
print("=" * 60)

SCENARIOS = [
    ("A2", 1),
    ("A2", -1),
    ("A8", 1),
    ("A8", -1),
    ("A13", 1),
    ("A13", -1),
]

HIGHWAY_CONFIG = {
    "A2": {"pk_start": 0, "pk_end": 291},
    "A8": {"pk_start": 0, "pk_end": 87},
    "A13": {"pk_start": 0, "pk_end": 190},
}

# Fixed number of cars per highway
FIXED_N_CARS_BY_HIGHWAY = {
    "A2": 798,
    "A8": 26,
    "A13": 258,
}


def n_cars_for_highway(highway):
    """
    Fixed demand per highway.
    The same number is used for both directions of each highway.
    """
    return FIXED_N_CARS_BY_HIGHWAY[highway]


# Important:
# N cars are injected during this fixed time window.
# This makes N represent traffic intensity, not simulation duration.
DEMAND_WINDOW_MIN = 60.0

SPEED_KMH = 100

BATTERY_MIN_PCT = 10.0
BATTERY_MAX_PCT = 90.0
BATTERY_CONSUMPTION_MIN_PCT_PER_KM = 0.15
BATTERY_CONSUMPTION_MAX_PCT_PER_KM = 0.25
FULL_BATTERY_PCT = 100.0

RECHARGE_TIME_MIN = 10

# Stations with confirmed presence but unknown charger count get this value.
DEFAULT_MIN_CHARGERS = 1

# If every reachable station has waiting time above this threshold,
# the car is rejected and the system is saturated.
WAITING_TIME_THRESHOLD_MIN = 15.0

print("Scénarios étudiés avec N fixe :")
for highway, sens in SCENARIOS:
    print(f" - {highway}, sens={sens} → N_CARS={n_cars_for_highway(highway)}")

print(f"\nFenêtre d'arrivée     : {DEMAND_WINDOW_MIN:.0f} min")
print("Toutes les voitures entrent pendant cette fenêtre.")
print(f"Temps de recharge     : {RECHARGE_TIME_MIN} min")
print(f"Batterie initiale     : entre {BATTERY_MIN_PCT:.0f}% et {BATTERY_MAX_PCT:.0f}%")
print(
    f"Conso batterie        : "
    f"{BATTERY_CONSUMPTION_MIN_PCT_PER_KM:.2f}% à "
    f"{BATTERY_CONSUMPTION_MAX_PCT_PER_KM:.2f}% par km"
)
print(f"Bornes min par défaut : {DEFAULT_MIN_CHARGERS}")
print(f"Seuil d'attente       : {WAITING_TIME_THRESHOLD_MIN} min")


# ============================================================
# Step 4 — Scenario function
# ============================================================

def run_scenario(
    selected_highway,
    selected_sens,
    scenario_id,
    n_cars_override=None,
    save_outputs=True,
    verbose=True,
):
    """
    Run one highway-direction scenario.

    System failure definition:
    - A car is physically feasible if it can reach at least one station and
      can reach the exit after charging.
    - A feasible car is accepted only if one reachable station has waiting time
      <= WAITING_TIME_THRESHOLD_MIN.
    - If at least one feasible car is rejected because of excessive waiting time,
      the system is saturated.
    """

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    log("\n" + "#" * 80)
    log(f"SCÉNARIO {scenario_id} — {selected_highway}, sens={selected_sens}")
    log("#" * 80)

    sens_label = "sens_plus" if selected_sens == 1 else "sens_minus"
    output_prefix = f"{selected_highway}_{sens_label}"

    if n_cars_override is None:
        N_CARS = n_cars_for_highway(selected_highway)
    else:
        N_CARS = int(n_cars_override)

    # ----------------------------------------------------------
    # Local rest-area copy with charger counts
    # ----------------------------------------------------------

    local_df_aires = df_aires.copy()
    local_df_aires["existing_chargers_model"] = 0

    if "known_chargers_comments" in local_df_aires.columns:
        local_df_aires["existing_chargers_model"] = (
            local_df_aires["known_chargers_comments"].fillna(0)
        )
    elif "existing_chargers" in local_df_aires.columns:
        local_df_aires["existing_chargers_model"] = (
            local_df_aires["existing_chargers"].fillna(0)
        )

    local_df_aires.loc[
        local_df_aires["has_existing_charger"]
        & (local_df_aires["existing_chargers_model"] <= 0),
        "existing_chargers_model",
    ] = DEFAULT_MIN_CHARGERS

    local_df_aires.loc[
        ~local_df_aires["has_existing_charger"],
        "existing_chargers_model",
    ] = 0

    local_df_aires["existing_chargers_model"] = (
        local_df_aires["existing_chargers_model"].astype(int)
    )

    # ----------------------------------------------------------
    # Select existing stations
    # ----------------------------------------------------------

    df_stations = local_df_aires[
        (local_df_aires["autoroute"] == selected_highway)
        & (local_df_aires["sens"] == selected_sens)
        & (local_df_aires["has_existing_charger"])
    ].copy()

    if df_stations.empty:
        log(
            f"⚠️ Aucune station existante trouvée pour "
            f"{selected_highway}, sens={selected_sens}."
        )

        return {
            "scenario": output_prefix,
            "highway": selected_highway,
            "sens": selected_sens,
            "status": "NO_STATION",
            "system_ok": False,
            "n_cars": N_CARS,
            "feasible_cars": 0,
            "infeasible_cars": N_CARS,
            "assigned_cars": 0,
            "rejected_waiting_time_cars": 0,
            "pct_rejected_waiting_time": None,
            "n_stations": 0,
            "total_waiting_time_min": None,
            "average_waiting_time_min": None,
            "max_waiting_time_min": None,
            "pct_assigned_furthest": None,
            "average_fallback_steps": None,
        }

    if selected_sens == 1:
        df_stations = df_stations.sort_values("pk").reset_index(drop=True)
    else:
        df_stations = df_stations.sort_values("pk", ascending=False).reset_index(drop=True)

    df_stations["station_id"] = range(len(df_stations))
    stations = df_stations.to_dict("records")
    m_stations = len(stations)

    log(f"\nStations existantes retenues : {m_stations}")

    if DISPLAY_TABLES and verbose:
        log(
            df_stations[
                [
                    "station_id",
                    "autoroute",
                    "nom",
                    "sens",
                    "pk",
                    "existing_chargers_model",
                    "Data status",
                ]
            ].to_string(index=False)
        )

    # ----------------------------------------------------------
    # Entry and exit PK
    # ----------------------------------------------------------

    cfg = HIGHWAY_CONFIG[selected_highway]

    if selected_sens == 1:
        entry_pk = float(cfg["pk_start"])
        exit_pk = float(cfg["pk_end"])
    else:
        entry_pk = float(cfg["pk_end"])
        exit_pk = float(cfg["pk_start"])

    axis_length_km = abs(exit_pk - entry_pk)

    log(f"\nPK entrée : {entry_pk:.1f} km")
    log(f"PK sortie : {exit_pk:.1f} km")
    log(f"Longueur de l'axe : {axis_length_km:.1f} km")
    log(f"Nombre de voitures : {N_CARS}")
    log(f"Fenêtre d'arrivée : {DEMAND_WINDOW_MIN:.1f} min")

    # ----------------------------------------------------------
    # Generate EV cars
    # ----------------------------------------------------------

    rng = np.random.default_rng(42 + scenario_id)

    battery_levels_pct = rng.uniform(
        low=BATTERY_MIN_PCT,
        high=BATTERY_MAX_PCT,
        size=N_CARS,
    )

    consumption_pct_per_km = rng.uniform(
        low=BATTERY_CONSUMPTION_MIN_PCT_PER_KM,
        high=BATTERY_CONSUMPTION_MAX_PCT_PER_KM,
        size=N_CARS,
    )

    cars = []

    for car_id in range(N_CARS):
        if N_CARS == 1:
            entry_time_min = 0.0
        else:
            entry_time_min = car_id * DEMAND_WINDOW_MIN / (N_CARS - 1)

        battery_pct = float(battery_levels_pct[car_id])
        consumption = float(consumption_pct_per_km[car_id])
        remaining_range_km = battery_pct / consumption

        cars.append(
            {
                "car_id": car_id,
                "entry_time_min": entry_time_min,
                "entry_pk": entry_pk,
                "exit_pk": exit_pk,
                "battery_pct_initial": battery_pct,
                "consumption_pct_per_km": consumption,
                "remaining_range_km": remaining_range_km,
            }
        )

    df_cars = pd.DataFrame(cars)

    if save_outputs:
        cars_output_path = os.path.join(
            OUTPUT_DIR,
            f"{output_prefix}_generated_cars.csv",
        )

        df_cars.to_csv(
            cars_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    log("\nAperçu voitures :")
    log(df_cars.head(5).to_string(index=False))

    if len(df_cars) > 5:
        log(f"... {len(df_cars) - 5} autres voitures non affichées")

    # ----------------------------------------------------------
    # Reachability
    # ----------------------------------------------------------

    reachability = {}
    feasible_stations_by_car = defaultdict(list)

    for car in cars:
        car_id = car["car_id"]
        entry_time_min = car["entry_time_min"]
        battery_pct = car["battery_pct_initial"]
        consumption = car["consumption_pct_per_km"]

        for st in stations:
            station_id = st["station_id"]
            station_pk = float(st["pk"])

            if selected_sens == 1:
                distance_to_station = station_pk - entry_pk
                station_after_entry = station_pk >= entry_pk
                distance_station_to_exit = exit_pk - station_pk
            else:
                distance_to_station = entry_pk - station_pk
                station_after_entry = station_pk <= entry_pk
                distance_station_to_exit = station_pk - exit_pk

            if not station_after_entry:
                continue

            battery_used_to_station_pct = distance_to_station * consumption
            battery_pct_at_station = battery_pct - battery_used_to_station_pct

            can_reach_station = battery_pct_at_station >= 0

            battery_needed_to_exit_pct = distance_station_to_exit * consumption
            can_reach_exit_after_charge = battery_needed_to_exit_pct <= FULL_BATTERY_PCT

            if can_reach_station and can_reach_exit_after_charge:
                travel_time_min = distance_to_station / SPEED_KMH * 60
                arrival_time_min = entry_time_min + travel_time_min
                slot = int(arrival_time_min // RECHARGE_TIME_MIN)

                reachability[(car_id, station_id)] = {
                    "distance_to_station_km": distance_to_station,
                    "distance_station_to_exit_km": distance_station_to_exit,
                    "arrival_time_min": arrival_time_min,
                    "slot": slot,
                    "battery_used_to_station_pct": battery_used_to_station_pct,
                    "battery_pct_at_station": battery_pct_at_station,
                    "battery_needed_to_exit_after_charge_pct": battery_needed_to_exit_pct,
                }

                feasible_stations_by_car[car_id].append(station_id)

    infeasible_cars = [
        car["car_id"]
        for car in cars
        if len(feasible_stations_by_car[car["car_id"]]) == 0
    ]

    feasible_cars = [
        car
        for car in cars
        if car["car_id"] not in infeasible_cars
    ]

    log(f"\nVoitures totales                      : {N_CARS}")
    log(f"Voitures avec au moins une station OK : {len(feasible_cars)}")
    log(f"Voitures impossibles physiquement     : {len(infeasible_cars)}")

    reachability_matrix = np.zeros((N_CARS, m_stations), dtype=int)

    for car_id, station_id in reachability:
        reachability_matrix[car_id, station_id] = 1

    df_reachability = pd.DataFrame(
        reachability_matrix,
        columns=[f"station_{s['station_id']}_{s['nom']}" for s in stations],
    )

    df_reachability.insert(0, "car_id", range(N_CARS))

    if save_outputs:
        reachability_output_path = os.path.join(
            OUTPUT_DIR,
            f"{output_prefix}_reachability_matrix_cars_stations.csv",
        )

        df_reachability.to_csv(
            reachability_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    if len(feasible_cars) == 0:
        log("⚠️ Aucune voiture physiquement faisable dans ce scénario.")

        return {
            "scenario": output_prefix,
            "highway": selected_highway,
            "sens": selected_sens,
            "status": "NO_FEASIBLE_CAR",
            "system_ok": False,
            "n_cars": N_CARS,
            "feasible_cars": 0,
            "infeasible_cars": len(infeasible_cars),
            "assigned_cars": 0,
            "rejected_waiting_time_cars": 0,
            "pct_rejected_waiting_time": None,
            "n_stations": m_stations,
            "total_waiting_time_min": None,
            "average_waiting_time_min": None,
            "max_waiting_time_min": None,
            "pct_assigned_furthest": None,
            "average_fallback_steps": None,
        }

    # ----------------------------------------------------------
    # Sequential assignment with saturation rule
    # ----------------------------------------------------------

    station_charger_available_times = {
        st["station_id"]: [0.0] * int(st["existing_chargers_model"])
        for st in stations
    }

    assignments = []
    rejected_assignments = []

    feasible_cars_sorted = sorted(
        feasible_cars,
        key=lambda c: c["entry_time_min"],
    )

    for car in feasible_cars_sorted:
        car_id = car["car_id"]
        feasible_list = feasible_stations_by_car[car_id]
        last_rank = len(feasible_list) - 1

        candidate_records = []
        chosen = None

        # Try furthest station first, then fallback backward.
        for rank in range(last_rank, -1, -1):
            station_id = feasible_list[rank]
            info = reachability[(car_id, station_id)]

            arrival = float(info["arrival_time_min"])
            avail = station_charger_available_times[station_id]

            next_charger_idx = int(np.argmin(avail))
            expected_start = max(arrival, avail[next_charger_idx])
            expected_wait = expected_start - arrival

            candidate_records.append(
                {
                    "station_id": station_id,
                    "rank": rank,
                    "arrival": arrival,
                    "next_charger_idx": next_charger_idx,
                    "expected_start": expected_start,
                    "expected_wait": expected_wait,
                }
            )

            if expected_wait <= WAITING_TIME_THRESHOLD_MIN:
                chosen = candidate_records[-1]
                break

        # If all reachable stations exceed the threshold, reject the car.
        if chosen is None:
            best_candidate = min(
                candidate_records,
                key=lambda r: (r["expected_wait"], -r["rank"]),
            )

            rejected_assignments.append(
                {
                    "scenario": output_prefix,
                    "car_id": car_id,
                    "highway": selected_highway,
                    "sens": selected_sens,
                    "entry_time_min": car["entry_time_min"],
                    "battery_pct_initial": car["battery_pct_initial"],
                    "consumption_pct_per_km": car["consumption_pct_per_km"],
                    "remaining_range_km": car["remaining_range_km"],
                    "reason": "WAITING_TIME_ABOVE_THRESHOLD",
                    "best_station_id": best_candidate["station_id"],
                    "best_possible_waiting_time_min": best_candidate["expected_wait"],
                    "waiting_time_threshold_min": WAITING_TIME_THRESHOLD_MIN,
                }
            )

            continue

        station_id = chosen["station_id"]
        rank = chosen["rank"]

        station = stations[station_id]
        info = reachability[(car_id, station_id)]

        arrival = chosen["arrival"]
        charger_idx = chosen["next_charger_idx"]
        charging_start = chosen["expected_start"]
        waiting_time = chosen["expected_wait"]
        charging_end = charging_start + RECHARGE_TIME_MIN

        station_charger_available_times[station_id][charger_idx] = charging_end

        assignments.append(
            {
                "scenario": output_prefix,
                "car_id": car_id,
                "highway": selected_highway,
                "sens": selected_sens,
                "entry_time_min": car["entry_time_min"],
                "battery_pct_initial": car["battery_pct_initial"],
                "consumption_pct_per_km": car["consumption_pct_per_km"],
                "remaining_range_km": car["remaining_range_km"],
                "assigned_station_id": station_id,
                "assigned_station_name": station["nom"],
                "station_pk": station["pk"],
                "station_chargers": station["existing_chargers_model"],
                "arrival_time_min": arrival,
                "arrival_slot": info["slot"],
                "waiting_time_min": waiting_time,
                "charging_start_min": charging_start,
                "charging_end_min": charging_end,
                "assigned_charger_index": charger_idx,
                "distance_to_station_km": info["distance_to_station_km"],
                "distance_station_to_exit_km": info["distance_station_to_exit_km"],
                "battery_used_to_station_pct": info["battery_used_to_station_pct"],
                "battery_pct_at_station": info["battery_pct_at_station"],
                "battery_needed_to_exit_after_charge_pct": info[
                    "battery_needed_to_exit_after_charge_pct"
                ],
                "battery_pct_after_charge": FULL_BATTERY_PCT,
                "station_rank_assigned": rank,
                "station_rank_max_feasible": last_rank,
                "assigned_furthest": int(rank == last_rank),
                "fallback_steps": last_rank - rank,
            }
        )

    df_assignments = pd.DataFrame(assignments)
    df_rejected = pd.DataFrame(rejected_assignments)

    n_assigned = len(df_assignments)
    n_rejected_wait = len(df_rejected)

    system_ok = n_rejected_wait == 0

    if n_assigned > 0:
        total_wait = df_assignments["waiting_time_min"].sum()
        average_wait = df_assignments["waiting_time_min"].mean()
        max_wait = df_assignments["waiting_time_min"].max()
        pct_furthest = df_assignments["assigned_furthest"].mean() * 100
        avg_fallback_steps = df_assignments["fallback_steps"].mean()
    else:
        total_wait = 0.0
        average_wait = None
        max_wait = None
        pct_furthest = None
        avg_fallback_steps = None

    pct_rejected_wait = (
        n_rejected_wait / len(feasible_cars) * 100
        if len(feasible_cars) > 0
        else 0.0
    )

    log(f"\nSystème OK selon le seuil d'attente : {system_ok}")
    log(f"Voitures assignées                 : {n_assigned}")
    log(f"Voitures rejetées par saturation    : {n_rejected_wait}")
    log(f"% rejetées par saturation           : {pct_rejected_wait:.1f} %")

    if n_assigned > 0:
        log(f"Temps d'attente total               : {total_wait:.2f} min")
        log(f"Temps d'attente moyen               : {average_wait:.2f} min")
        log(f"Temps d'attente maximum             : {max_wait:.2f} min")
        log(f"% voitures à la station la plus loin: {pct_furthest:.1f} %")
        log(f"Recul moyen m -> m-k                : {avg_fallback_steps:.2f} station(s)")

    # ----------------------------------------------------------
    # Station summary
    # ----------------------------------------------------------

    station_summary_rows = []

    for st in stations:
        sid = st["station_id"]

        if n_assigned > 0:
            sub = df_assignments[df_assignments["assigned_station_id"] == sid]
        else:
            sub = pd.DataFrame()

        station_summary_rows.append(
            {
                "scenario": output_prefix,
                "highway": selected_highway,
                "sens": selected_sens,
                "station_id": sid,
                "station_name": st["nom"],
                "pk": st["pk"],
                "chargers": st["existing_chargers_model"],
                "data_status": st.get("Data status", ""),
                "assigned_cars": len(sub),
                "total_waiting_time_min": (
                    sub["waiting_time_min"].sum()
                    if len(sub) > 0
                    else 0.0
                ),
                "average_waiting_time_min": (
                    sub["waiting_time_min"].mean()
                    if len(sub) > 0
                    else 0.0
                ),
                "max_waiting_time_min": (
                    sub["waiting_time_min"].max()
                    if len(sub) > 0
                    else 0.0
                ),
            }
        )

    df_station_summary = pd.DataFrame(station_summary_rows)

    log("\nRésumé par station : sauvegardé en CSV, non affiché dans le terminal.")

    if DISPLAY_TABLES and verbose:
        log(df_station_summary.to_string(index=False))
    elif verbose:
        busiest = df_station_summary.sort_values(
            ["assigned_cars", "max_waiting_time_min"],
            ascending=False,
        ).head(5)

        log("Top 5 stations les plus utilisées :")
        log(
            busiest[
                [
                    "station_name",
                    "assigned_cars",
                    "average_waiting_time_min",
                    "max_waiting_time_min",
                ]
            ].to_string(index=False)
        )

    # ----------------------------------------------------------
    # Decision matrix
    # ----------------------------------------------------------

    decision_matrix = np.zeros((N_CARS, m_stations), dtype=int)

    if n_assigned > 0:
        for _, row in df_assignments.iterrows():
            decision_matrix[
                int(row["car_id"]),
                int(row["assigned_station_id"]),
            ] = 1

    df_decision_matrix = pd.DataFrame(
        decision_matrix,
        columns=[f"station_{s['station_id']}_{s['nom']}" for s in stations],
    )

    df_decision_matrix.insert(0, "car_id", range(N_CARS))

    # ----------------------------------------------------------
    # Save files
    # ----------------------------------------------------------

    if save_outputs:
        files_to_save = [
            (
                df_assignments,
                f"{output_prefix}_car_station_assignments.csv",
            ),
            (
                df_station_summary,
                f"{output_prefix}_station_queue_summary.csv",
            ),
            (
                df_decision_matrix,
                f"{output_prefix}_decision_matrix_car_station.csv",
            ),
            (
                df_rejected,
                f"{output_prefix}_rejected_cars.csv",
            ),
        ]

        for df_to_save, fname in files_to_save:
            path = os.path.join(OUTPUT_DIR, fname)
            df_to_save.to_csv(path, index=False, encoding="utf-8-sig")

        log("\n✅ Fichiers sauvegardés :")
        log(f" - Outputs/{output_prefix}_generated_cars.csv")
        log(f" - Outputs/{output_prefix}_reachability_matrix_cars_stations.csv")
        log(f" - Outputs/{output_prefix}_car_station_assignments.csv")
        log(f" - Outputs/{output_prefix}_station_queue_summary.csv")
        log(f" - Outputs/{output_prefix}_decision_matrix_car_station.csv")
        log(f" - Outputs/{output_prefix}_rejected_cars.csv")

    return {
        "scenario": output_prefix,
        "highway": selected_highway,
        "sens": selected_sens,
        "status": "SYSTEM_OK" if system_ok else "SYSTEM_SATURATED",
        "system_ok": system_ok,
        "n_cars": N_CARS,
        "feasible_cars": len(feasible_cars),
        "infeasible_cars": len(infeasible_cars),
        "assigned_cars": n_assigned,
        "rejected_waiting_time_cars": n_rejected_wait,
        "pct_rejected_waiting_time": pct_rejected_wait,
        "n_stations": m_stations,
        "total_waiting_time_min": total_wait,
        "average_waiting_time_min": average_wait,
        "max_waiting_time_min": max_wait,
        "pct_assigned_furthest": pct_furthest,
        "average_fallback_steps": avg_fallback_steps,
    }


# ============================================================
# Step 4B — Find maximum N before saturation
# ============================================================

def find_breaking_n_for_scenario(
    selected_highway,
    selected_sens,
    scenario_id,
    n_min=1,
    n_max_initial=100,
    n_max_limit=100_000,
):
    """
    Find the largest N for which the system still works.

    The search has two phases:
    1. Increase N until the system saturates.
    2. Binary search between the last OK N and the first saturated N.
    """

    print("\n" + "=" * 80)
    print(f"RECHERCHE DU N CRITIQUE — {selected_highway}, sens={selected_sens}")
    print("=" * 80)

    result_min = run_scenario(
        selected_highway=selected_highway,
        selected_sens=selected_sens,
        scenario_id=scenario_id,
        n_cars_override=n_min,
        save_outputs=False,
        verbose=False,
    )

    if not result_min["system_ok"]:
        print(f"⚠️ Le système sature déjà à N={n_min}.")

        return {
            "highway": selected_highway,
            "sens": selected_sens,
            "status": "BREAK_AT_MIN",
            "max_working_n": 0,
            "first_failing_n": n_min,
            "threshold_waiting_time_min": WAITING_TIME_THRESHOLD_MIN,
            "demand_window_min": DEMAND_WINDOW_MIN,
            "last_ok_assigned_cars": 0,
            "first_bad_assigned_cars": result_min["assigned_cars"],
            "first_bad_rejected_cars": result_min["rejected_waiting_time_cars"],
            "first_bad_pct_rejected": result_min["pct_rejected_waiting_time"],
        }

    low = n_min
    high = n_max_initial

    last_ok_result = result_min
    first_bad_result = None

    # Phase 1: find an upper bound that saturates
    while high <= n_max_limit:
        result = run_scenario(
            selected_highway=selected_highway,
            selected_sens=selected_sens,
            scenario_id=scenario_id,
            n_cars_override=high,
            save_outputs=False,
            verbose=False,
        )

        print(
            f"Test N={high:<8} → "
            f"{'OK' if result['system_ok'] else 'SATURÉ'} "
            f"| assignées={result['assigned_cars']} "
            f"| rejetées={result['rejected_waiting_time_cars']}"
        )

        if result["system_ok"]:
            low = high
            last_ok_result = result
            high *= 2
        else:
            first_bad_result = result
            break

    if first_bad_result is None:
        print(
            f"⚠️ Aucun point de saturation trouvé jusqu'à N={n_max_limit}. "
            f"Augmente n_max_limit ou réduis DEMAND_WINDOW_MIN."
        )

        return {
            "highway": selected_highway,
            "sens": selected_sens,
            "status": "NO_BREAK_FOUND",
            "max_working_n": low,
            "first_failing_n": None,
            "threshold_waiting_time_min": WAITING_TIME_THRESHOLD_MIN,
            "demand_window_min": DEMAND_WINDOW_MIN,
            "last_ok_assigned_cars": last_ok_result["assigned_cars"],
            "first_bad_assigned_cars": None,
            "first_bad_rejected_cars": None,
            "first_bad_pct_rejected": None,
        }

    # Phase 2: binary search
    left = low
    right = high

    while right - left > 1:
        mid = (left + right) // 2

        result = run_scenario(
            selected_highway=selected_highway,
            selected_sens=selected_sens,
            scenario_id=scenario_id,
            n_cars_override=mid,
            save_outputs=False,
            verbose=False,
        )

        print(
            f"Recherche binaire N={mid:<8} → "
            f"{'OK' if result['system_ok'] else 'SATURÉ'} "
            f"| assignées={result['assigned_cars']} "
            f"| rejetées={result['rejected_waiting_time_cars']}"
        )

        if result["system_ok"]:
            left = mid
            last_ok_result = result
        else:
            right = mid
            first_bad_result = result

    print(f"\nN maximum qui fonctionne : {left}")
    print(f"Premier N qui sature     : {right}")

    return {
        "highway": selected_highway,
        "sens": selected_sens,
        "status": "BREAK_FOUND",
        "max_working_n": left,
        "first_failing_n": right,
        "threshold_waiting_time_min": WAITING_TIME_THRESHOLD_MIN,
        "demand_window_min": DEMAND_WINDOW_MIN,
        "last_ok_assigned_cars": (
            last_ok_result["assigned_cars"]
            if last_ok_result is not None
            else None
        ),
        "first_bad_assigned_cars": (
            first_bad_result["assigned_cars"]
            if first_bad_result is not None
            else None
        ),
        "first_bad_rejected_cars": (
            first_bad_result["rejected_waiting_time_cars"]
            if first_bad_result is not None
            else None
        ),
        "first_bad_pct_rejected": (
            first_bad_result["pct_rejected_waiting_time"]
            if first_bad_result is not None
            else None
        ),
    }

# ============================================================
# Step 5 — Run fixed-N scenarios
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 — Lancement des scénarios avec N fixe")
print("=" * 60)

fixed_n_summary = []

for scenario_id, (highway, sens) in enumerate(SCENARIOS):
    N_CARS = n_cars_for_highway(highway)

    result = run_scenario(
        selected_highway=highway,
        selected_sens=sens,
        scenario_id=scenario_id,
        n_cars_override=N_CARS,
        save_outputs=True,
        verbose=True,
    )

    fixed_n_summary.append(result)

df_fixed_n_summary = pd.DataFrame(fixed_n_summary)

fixed_n_summary_path = os.path.join(
    OUTPUT_DIR,
    "fixed_n_scenario_summary.csv",
)

df_fixed_n_summary.to_csv(
    fixed_n_summary_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 80)
print("RÉSUMÉ DES SCÉNARIOS AVEC N FIXE")
print("=" * 80)
print(df_fixed_n_summary.to_string(index=False))

print(f"\nRésumé sauvegardé : {fixed_n_summary_path}")

# ============================================================
# Interpretation
# ============================================================

print("\n" + "=" * 80)
print("INTERPRÉTATION")
print("=" * 80)

print(
    """
Le système est considéré comme fonctionnel tant que chaque voiture physiquement
faisable peut être affectée à une station atteignable avec un temps d'attente
inférieur ou égal au seuil fixé.

Le système est considéré comme saturé dès qu'au moins une voiture physiquement
faisable ne trouve aucune station atteignable avec une attente acceptable.

La variable N représente maintenant une intensité de demande :
toutes les voitures entrent dans une fenêtre fixe de temps.

Fichiers principaux produits :
- Outputs/critical_n_summary.csv
  → N maximum qui fonctionne et premier N qui sature.

- Outputs/first_failing_scenarios_summary.csv
  → résumé détaillé des scénarios au premier N saturé.

- Outputs/*_rejected_cars.csv
  → voitures rejetées parce que toutes les stations atteignables dépassaient
    le seuil d'attente.

- Outputs/*_car_station_assignments.csv
  → voitures effectivement affectées à une station.

- Outputs/*_station_queue_summary.csv
  → résumé des files d'attente par station.
"""
)