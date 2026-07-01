import pandas as pd

# The rdered list of all config keys.
# Order matters 
CONFIG_KEYS = [
    "File_Chunk_No",
    "Frame_Length",
    "Downsample_fs",
    "Dynamic_Range_Lower",
    "Dynamic_Range_Upper",
    "CP_WindowSize",
    "CP_Overlap",
    "CP_F1",
    "CP_F2",
    "SA_WindowSize",
    "SA_Overlap",
    "SA_F1",
    "SA_F2",
    "SB_WindowSize",
    "SB_Overlap",
    "SB_F1",
    "SB_F2",
    "SC_WindowSize",
    "SC_Overlap",
    "SC_F1",
    "SC_F2",
] + [f"label_{i}" for i in range(1, 19)] + [
    "Annotator_ID",
]

# Keys whose values should be parsed as numbers.
# Label keys are always kept as strings.
NUMERIC_KEYS = set(CONFIG_KEYS[:21])


def load_config(filepath: str) -> dict:
    """
    Read a HydroSeek config CSV and return a dict of key/ value.

    Numeric keys are returned as int or float.
    Label keys and annotator id are returned as strings (empty string if blank).
    """
    df = pd.read_csv(filepath, dtype=str)

    if "Config_Features" not in df.columns or "Config_Settings" not in df.columns:
        raise ValueError(
            f"Config file must have columns 'Config_Features' and "
            f"'Config_Settings'. Got: {list(df.columns)}"
        )

    config = {}
    for _, row in df.iterrows():
        key = str(row["Config_Features"]).strip()
        val = str(row["Config_Settings"]).strip()

        if key in NUMERIC_KEYS:
            config[key] = _parse_number(val)
        else:
            # Label fields: empty string if the CSV cell was blank
            config[key] = "" if val in ("", "nan", "None") else val

    return config


def export_config(filepath: str, values: dict) -> None:
    """
    Write a HydroSeek config dict to a CSV file.

    Keys are written in the canonical CONFIG_KEYS order so the output
    file always matches the format of the supplied hydroseek_config.csv.

    """
    rows = []
    for key in CONFIG_KEYS:
        raw = values.get(key, "")
        rows.append({
            "Config_Features": key,
            "Config_Settings": "" if raw is None else str(raw),
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"Config saved to {filepath}")


def config_to_state_kwargs(config: dict) -> dict:
    """
    Convert a loaded config dict into keyword arguments that can be
    passed directly to AppState fields.
    """
    labels = [config.get(f"label_{i}", "") for i in range(1, 19)]

    return {
        "file_chunk_number":  int(config.get("File_Chunk_No",  1)),
        "label_frame_length": float(config.get("Frame_Length",   5.0)),
        "target_fs":          int(config.get("Downsample_fs",  5000)),
        "dynamic_range_l":    float(config.get("Dynamic_Range_Lower", -80.0)),
        "dynamic_range_u":    float(config.get("Dynamic_Range_Upper",  10.0)),
        "windowsize_cp":      int(config.get("CP_WindowSize",   2048)),
        "overlap_cp":         float(config.get("CP_Overlap",     50.0)),
        "min_f_cp":           float(config.get("CP_F1",          10.0)),
        "max_f_cp":           float(config.get("CP_F2",       48000.0)),
        "windowsize_sa":      int(config.get("SA_WindowSize",   2048)),
        "overlap_sa":         float(config.get("SA_Overlap",     75.0)),
        "min_f_sa":           float(config.get("SA_F1",          10.0)),
        "max_f_sa":           float(config.get("SA_F2",       48000.0)),
        "windowsize_sb":      int(config.get("SB_WindowSize",   1024)),
        "overlap_sb":         float(config.get("SB_Overlap",     50.0)),
        "min_f_sb":           float(config.get("SB_F1",          10.0)),
        "max_f_sb":           float(config.get("SB_F2",       20000.0)),
        "windowsize_sc":      int(config.get("SC_WindowSize",    512)),
        "overlap_sc":         float(config.get("SC_Overlap",     75.0)),
        "min_f_sc":           float(config.get("SC_F1",          10.0)),
        "max_f_sc":           float(config.get("SC_F2",        2000.0)),
        "labels":             labels,
        "annotator_id":       config.get("Annotator_ID", ""),
    }



# Internal helpers                                                     

def _parse_number(value: str):
    """
    Parse a string as int if it looks like a whole number,
    otherwise as float. Returns 0 if parsing fails.
    """
    try:
        f = float(value)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return 0