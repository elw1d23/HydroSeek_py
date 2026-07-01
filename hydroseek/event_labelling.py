"""
event_labelling.py — data layer for HydroSeek event annotations.

Handles point labels and bounding-box (rectangle) annotations placed
directly on the Context, A, B and C spectrogram canvases.

Kept intentionally separate from labelling.py so the two output tables
(frame labels and event labels) have independent modules.

create_event_labels_table()     -> pd.DataFrame  9empty)
append_event(...)               -> pd.DataFrame  (new row added)
remove_last_event(...)          -> pd.DataFrame  (last row dropped)
get_events_for_frame(...)       -> pd.DataFrame  (filtered by ChunkNo)
export_event_labels(...)        -> str            (path written)
export_event_config(...)        -> str            (path written)
"""

import os
import pandas as pd


# Column order for the event labels CSV.
EVENT_COLUMNS = [
    "Event_ID",
    "Type",
    "Plot",
    "ChunkIdx",
    "FrameNo",
    "Frame_StartTime_sec",
    "Frame_EndTime_sec",
    "Event_Time_sec",
    "Event_Time_End_sec",
    "Freq_Hz_low",
    "Freq_Hz_high",
    "Event_Label",
    "Confidence",
    "Comment",
]

# Column order for the config CSV.
CONFIG_COLUMNS = ["Parameter", "Value"]


def create_event_labels_table() -> pd.DataFrame:
    """
    Return an empty DataFrame with the correct event-label column schema.

    Called once at session start (in setup_tab._load_audio_and_initialise)
    and stored in AppState.event_labels_table.
    """
    return pd.DataFrame(columns=EVENT_COLUMNS)


def append_event(
    table: pd.DataFrame,
    event_id: int,
    event_type: str,
    plot: str,
    chunk_idx: int,
    frame_no: int,
    frame_start: float,
    frame_end: float,
    event_time: float,
    event_time_end: float | None,
    freq_low: float,
    freq_high: float | None,
    label: str,
    confidence: int,
    comment: str,
) -> pd.DataFrame:
    """
    Append one event row and return the updated DataFrame.

    Parameters
    
    event_id        : globally unique integer for this session (from AppState)
    event_type      : "point" or "box"
    plot            : "Context", "A", "B", or "C"
    chunk_idx   : zero-based index of the large audio chunk (resets to 0
                      on each new chunk).  Combined with chunk_no this gives a
                      unique key across the whole session.
    frame_no        : frame index within the current big chunk
    frame_start     : absolute start time of the frame in seconds
    frame_end       : absolute end time of the frame in seconds
    event_time      : click time (point) or box left edge (box), in seconds
                      relative to the axis displayed on that canvas
    event_time_end  : box right edge in seconds; None / NaN for point labels
    freq_low        : click frequency (point) or box lower freq (box), in Hz
    freq_high       : box upper frequency in Hz; None / NaN for point labels
    label           : free-text label entered by the user
    comment         : contents of the shared Notes field at time of placement
    """
    row = {
        "Event_ID":            event_id,
        "Type":                event_type,
        "Plot":                plot,
        "ChunkIdx":         chunk_idx,
        "FrameNo":             frame_no,
        "Frame_StartTime_sec": frame_start,
        "Frame_EndTime_sec":   frame_end,
        "Event_Time_sec":      event_time,
        "Event_Time_End_sec":  event_time_end if event_time_end is not None else float("nan"),
        "Freq_Hz_low":         freq_low,
        "Freq_Hz_high":        freq_high if freq_high is not None else float("nan"),
        "Event_Label":         label,
        "Comment":             comment,
    }
    new_row = pd.DataFrame([row], columns=EVENT_COLUMNS)
    return pd.concat([table, new_row], ignore_index=True)


def remove_last_event(table: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the most recently appended event row.

    """
    if len(table) == 0:
        return table
    return table.iloc[:-1].reset_index(drop=True)


def get_events_for_frame(
    table: pd.DataFrame, big_chunk_idx: int, chunk_no: int
) -> list[dict]:
    """
    Return all events belonging to the given frame as a list of dicts.

    Uses the compound key (ChunkIdx, FrameNo) so that frame 3 in chunk 2
    is never confused with frame 3 in chunk 1.

    Used by LabellingTab to redraw annotations when a frame
    is rendered (including on moving backwards).
    """
    if len(table) == 0:
        return []
    mask = (table["ChunkIdx"] == big_chunk_idx) & (table["FrameNo"] == chunk_no)
    return table[mask].to_dict(orient="records")


def export_event_labels(table: pd.DataFrame, audio_filepath: str, annotator_id: str = "",) -> str:
    """
    Writeevent labels table to <stem>_Event_labels{suffix}.csv alongside the
    audio file
    """
    directory = os.path.dirname(os.path.abspath(audio_filepath))
    stem      = os.path.splitext(os.path.basename(audio_filepath))[0]
    suffix    = f"_{annotator_id.strip()}" if annotator_id and annotator_id.strip() else ""
    csv_path  = os.path.join(directory, f"{stem}_Event_labels{suffix}.csv")
    table.to_csv(csv_path, index=False)
    print(f"Event labels written to {csv_path}")
    return csv_path


def export_event_config(state, audio_filepath: str,annotator_id: str = "",) -> str:
    """
    Write spectrogram config for the four labelling plots (Context, A, B, C)
    to <stem>_Event_labels_config.csv alongside the audio file.

    The config is written once at session start so the exact spectrogram
    settings used during annotation can always be reconstructed later.

    
    """
    directory = os.path.dirname(os.path.abspath(audio_filepath))
    stem      = os.path.splitext(os.path.basename(audio_filepath))[0]
    suffix    = f"_{annotator_id.strip()}" if annotator_id and annotator_id.strip() else ""
    csv_path  = os.path.join(directory, f"{stem}_Event_labels_config{suffix}.csv")

    rows = [
        ("CP_WindowSize",  state.windowsize_cp),
        ("CP_Overlap",     state.overlap_cp),
        ("CP_F1",          state.min_f_cp),
        ("CP_F2",          state.max_f_cp),
        ("SA_WindowSize",  state.windowsize_sa),
        ("SA_Overlap",     state.overlap_sa),
        ("SA_F1",          state.min_f_sa),
        ("SA_F2",          state.max_f_sa),
        ("SA_Mel",         int(state.mel_sa)),
        ("SB_WindowSize",  state.windowsize_sb),
        ("SB_Overlap",     state.overlap_sb),
        ("SB_F1",          state.min_f_sb),
        ("SB_F2",          state.max_f_sb),
        ("SB_Mel",         int(state.mel_sb)),
        ("SC_WindowSize",  state.windowsize_sc),
        ("SC_Overlap",     state.overlap_sc),
        ("SC_F1",          state.min_f_sc),
        ("SC_F2",          state.max_f_sc),
    ]

    cfg = pd.DataFrame(rows, columns=CONFIG_COLUMNS)
    cfg.to_csv(csv_path, index=False)
    print(f"Event config written to {csv_path}")
    return csv_path
