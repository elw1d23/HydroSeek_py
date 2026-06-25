import os
import numpy as np
import pandas as pd


# The three metadata columns always appear before the label columns
METADATA_COLUMNS = ["ChunkNo", "StartTime_sec", "EndTime_sec"]
# Suffix columns appended after the label columns — Count is NaN by default
SUFFIX_COLUMNS   = ["Confidence", "Comment", "Count"]


def create_labels_table(labels: list[str]) -> pd.DataFrame:
    """
    Initialise an empty DataFrame with the correct column structure.

    Equivalent to MATLAB LabelTableCreator nested function.

    Empty label strings are replaced with NA_1, NA_2, etc. so the
    CSV always has meaningful column headers even if the user left
    some label fields blank during setup.

    Parameters
    ----------
    labels : list of up to 18 label strings from the setup UI

    Returns
    -------
    Empty DataFrame with columns:
        ChunkNo, StartTime_sec, EndTime_sec,
        <label_1>, ..., <label_18>,
        Confidence, Comment, Count
    """
    sanitised = _sanitise_labels(labels)
    columns   = METADATA_COLUMNS + sanitised + SUFFIX_COLUMNS
    print(f"Labelling table initialised with {len(columns)} columns.")
    return pd.DataFrame(columns=columns)


def append_row(
    table: pd.DataFrame,
    chunk_no: int,
    start_time: float,
    end_time: float,
    checkbox_values: list[int],
    confidence: int,
    comment: str,
    count: float = float("nan"),
) -> pd.DataFrame:
    """
    Add a single labelled frame to the table.

    Called once per frame, immediately after the user clicks
    'Load Next Frame'.  The checkbox_values list must be the same
    length as the number of label columns in the table (up to 18),
    with 1 for checked and 0 for unchecked.

    count is NaN if the user never touched the counter on this frame,
    or an integer value if + or - was pressed at least once — matching
    the MATLAB CounterValue logic exactly.

    Returns a new DataFrame (pandas concat creates a new object,
    it does not modify in place).
    """
    n_label_cols = len(table.columns) - len(METADATA_COLUMNS) - len(SUFFIX_COLUMNS)

    if len(checkbox_values) != n_label_cols:
        raise ValueError(
            f"Expected {n_label_cols} checkbox values, got {len(checkbox_values)}"
        )

    row_data = [chunk_no, start_time, end_time] + checkbox_values + [confidence, comment, count]
    new_row  = pd.DataFrame([row_data], columns=table.columns)
    return pd.concat([table, new_row], ignore_index=True)


def remove_last_row(table: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the most recently added row.

    Called by the Previous button to undo the last label assignment
    before the user re-labels the frame.
    """
    if len(table) == 0:
        return table
    return table.iloc[:-1].reset_index(drop=True)


def fix_chunk_numbering(
    table: pd.DataFrame, frame_length_seconds: float
) -> pd.DataFrame:
    """
    Renumber ChunkNo sequentially and recalculate start/end times.

    Equivalent to MATLAB fixChunkNumbering nested function.

    The labelling loop increments a frame counter that resets on each
    big chunk, so the ChunkNo column collected during labelling is not
    a clean 1, 2, 3, ... sequence. This function corrects that at
    export time.

    Parameters
    ----------
    table                : the completed labels DataFrame
    frame_length_seconds : the label frame length set by the user (e.g. 5.0)
    """
    table = table.copy()
    n = len(table)

    table["ChunkNo"]       = np.arange(1, n + 1)
    table["StartTime_sec"] = np.arange(0, n) * frame_length_seconds
    table["EndTime_sec"]   = np.arange(1, n + 1) * frame_length_seconds

    return table


def export_labels(table: pd.DataFrame, audio_filepath: str) -> str:
    """
    Save the labels table as a CSV file alongside the audio file.

    The output filename is <audio_stem>_labels.csv, matching the
    MATLAB writetable call exactly. Returns the path written.
    """
    directory = os.path.dirname(os.path.abspath(audio_filepath))
    stem      = os.path.splitext(os.path.basename(audio_filepath))[0]
    csv_path  = os.path.join(directory, f"{stem}_labels.csv")
    table.to_csv(csv_path, index=False)
    print(f"Labels written to {csv_path}")
    return csv_path


# Internal helpers


def _sanitise_labels(labels: list[str]) -> list[str]:
    """Replace blank entries with NA_1, NA_2, etc."""
    result     = []
    na_counter = 1
    for lbl in labels:
        if not lbl or lbl.strip() == "":
            result.append(f"NA_{na_counter}")
            na_counter += 1
        else:
            result.append(lbl.strip())
    return result
