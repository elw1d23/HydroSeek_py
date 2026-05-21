import os
import numpy as np
import pandas as pd
import pytest

from hydroseek.labelling import (
    create_labels_table,
    append_row,
    remove_last_row,
    fix_chunk_numbering,
    export_labels,
)


# Fixtures                                                             


@pytest.fixture
def sample_labels():
    return ["Boat", "Rain", "Wind", "Fish", "", "", "", "",
            "", "", "", "", "", "", "", "", "", ""]


@pytest.fixture
def empty_table(sample_labels):
    return create_labels_table(sample_labels)


@pytest.fixture
def populated_table(empty_table):
    """A table with three rows already appended."""
    t = empty_table
    for i in range(3):
        checkboxes = [1, 0, 0, 0] + [0] * 14
        t = append_row(t, chunk_no=i + 1, start_time=i * 5.0,
                       end_time=(i + 1) * 5.0,
                       checkbox_values=checkboxes,
                       confidence=2, comment=f"frame {i}")
    return t


# create_labels_table                                                  


def test_column_count(empty_table):
    # 3 metadata + 18 labels + 2 suffix = 23 columns
    assert len(empty_table.columns) == 23


def test_metadata_columns_present(empty_table):
    for col in ["ChunkNo", "StartTime_sec", "EndTime_sec"]:
        assert col in empty_table.columns


def test_suffix_columns_present(empty_table):
    for col in ["Confidence", "Comment"]:
        assert col in empty_table.columns


def test_named_labels_used(empty_table):
    assert "Boat" in empty_table.columns
    assert "Rain" in empty_table.columns
    assert "Wind" in empty_table.columns


def test_blank_labels_become_na(empty_table):
    """Columns 5-18 were blank in sample_labels, should be NA_1..NA_14."""
    assert "NA_1" in empty_table.columns
    assert "NA_14" in empty_table.columns


def test_table_starts_empty(empty_table):
    assert len(empty_table) == 0


def test_all_18_label_columns_exist(sample_labels):
    """Even a fully named set of 18 labels should produce 18 label columns."""
    labels = [f"Label_{i}" for i in range(1, 19)]
    table  = create_labels_table(labels)
    label_cols = [c for c in table.columns
                  if c not in ["ChunkNo", "StartTime_sec", "EndTime_sec",
                               "Confidence", "Comment"]]
    assert len(label_cols) == 18


# append_row                                                           


def test_append_adds_one_row(empty_table):
    checkboxes = [1, 0, 1, 0] + [0] * 14
    result = append_row(empty_table, 1, 0.0, 5.0, checkboxes, 1, "test")
    assert len(result) == 1


def test_append_values_stored_correctly(empty_table):
    checkboxes = [1, 0, 1, 0] + [0] * 14
    result = append_row(empty_table, 1, 0.0, 5.0, checkboxes, 3, "hello")

    assert result.iloc[0]["ChunkNo"]       == 1
    assert result.iloc[0]["StartTime_sec"] == pytest.approx(0.0)
    assert result.iloc[0]["EndTime_sec"]   == pytest.approx(5.0)
    assert result.iloc[0]["Confidence"]    == 3
    assert result.iloc[0]["Comment"]       == "hello"
    assert result.iloc[0]["Boat"]          == 1
    assert result.iloc[0]["Wind"]          == 1


def test_append_multiple_rows(empty_table):
    t = empty_table
    for i in range(5):
        t = append_row(t, i + 1, i * 5.0, (i + 1) * 5.0,
                       [0] * 18, 1, "")
    assert len(t) == 5


def test_append_wrong_checkbox_count_raises(empty_table):
    with pytest.raises(ValueError):
        append_row(empty_table, 1, 0.0, 5.0, [1, 0], 1, "")



# remove_last_row                                                      


def test_remove_last_row(populated_table):
    result = remove_last_row(populated_table)
    assert len(result) == 2


def test_remove_last_row_on_empty_table(empty_table):
    result = remove_last_row(empty_table)
    assert len(result) == 0


def test_remove_preserves_earlier_rows(populated_table):
    result = remove_last_row(populated_table)
    assert result.iloc[0]["Comment"] == "frame 0"
    assert result.iloc[1]["Comment"] == "frame 1"


# fix_chunk_numbering                                                  


def test_chunk_numbers_sequential(populated_table):
    fixed = fix_chunk_numbering(populated_table, frame_length_seconds=5.0)
    expected = [1, 2, 3]
    assert list(fixed["ChunkNo"]) == expected


def test_start_times_correct(populated_table):
    fixed = fix_chunk_numbering(populated_table, frame_length_seconds=5.0)
    expected = [0.0, 5.0, 10.0]
    assert list(fixed["StartTime_sec"]) == pytest.approx(expected)


def test_end_times_correct(populated_table):
    fixed = fix_chunk_numbering(populated_table, frame_length_seconds=5.0)
    expected = [5.0, 10.0, 15.0]
    assert list(fixed["EndTime_sec"]) == pytest.approx(expected)


def test_fix_does_not_modify_original(populated_table):
    original_chunk_no = list(populated_table["ChunkNo"])
    fix_chunk_numbering(populated_table, frame_length_seconds=5.0)
    assert list(populated_table["ChunkNo"]) == original_chunk_no


# ------------------------------------------------------------------ #
# export_labels                                                        #
# ------------------------------------------------------------------ #

def test_export_creates_file(populated_table, tmp_path):
    audio_path = str(tmp_path / "recording.wav")
    csv_path   = export_labels(populated_table, audio_path)
    assert os.path.isfile(csv_path)


def test_export_filename_matches_audio(populated_table, tmp_path):
    audio_path = str(tmp_path / "recording.wav")
    csv_path   = export_labels(populated_table, audio_path)
    assert os.path.basename(csv_path) == "recording_labels.csv"


def test_export_csv_readable(populated_table, tmp_path):
    audio_path = str(tmp_path / "recording.wav")
    csv_path   = export_labels(populated_table, audio_path)
    loaded     = pd.read_csv(csv_path)
    assert len(loaded) == len(populated_table)
    assert list(loaded.columns) == list(populated_table.columns)