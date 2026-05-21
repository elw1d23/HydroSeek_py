import os
import pytest
import pandas as pd

from hydroseek.config import (
    load_config,
    export_config,
    config_to_state_kwargs,
    CONFIG_KEYS,
)


# Fixtures                                                             


@pytest.fixture
def sample_config_file(tmp_path):
    """Write a minimal but complete config CSV and return its path."""
    rows = []
    numeric_values = {
        "File_Chunk_No": "5", "Frame_Length": "5", "Downsample_fs": "5000",
        "Dynamic_Range_Lower": "-80", "Dynamic_Range_Upper": "10",
        "CP_WindowSize": "2048", "CP_Overlap": "50", "CP_F1": "10", "CP_F2": "48000",
        "SA_WindowSize": "2048", "SA_Overlap": "75", "SA_F1": "10", "SA_F2": "48000",
        "SB_WindowSize": "1024", "SB_Overlap": "50", "SB_F1": "10", "SB_F2": "20000",
        "SC_WindowSize": "512",  "SC_Overlap": "75", "SC_F1": "10", "SC_F2": "2000",
    }
    label_values = {f"label_{i}": chr(96 + i) for i in range(1, 19)}

    for key in CONFIG_KEYS:
        val = numeric_values.get(key) or label_values.get(key, "")
        rows.append({"Config_Features": key, "Config_Settings": val})

    path = tmp_path / "hydroseek_config.csv"
    pd.DataFrame(rows).to_csv(str(path), index=False)
    return str(path)


@pytest.fixture
def loaded_config(sample_config_file):
    return load_config(sample_config_file)

# load_config                                                          

def test_load_returns_dict(loaded_config):
    assert isinstance(loaded_config, dict)


def test_load_all_keys_present(loaded_config):
    for key in CONFIG_KEYS:
        assert key in loaded_config, f"Missing key: {key}"


def test_numeric_keys_are_numbers(loaded_config):
    assert isinstance(loaded_config["File_Chunk_No"],       int)
    assert isinstance(loaded_config["Frame_Length"],        (int, float))
    assert isinstance(loaded_config["Dynamic_Range_Lower"], (int, float))


def test_label_keys_are_strings(loaded_config):
    for i in range(1, 19):
        assert isinstance(loaded_config[f"label_{i}"], str)


def test_numeric_values_correct(loaded_config):
    assert loaded_config["File_Chunk_No"]       == 5
    assert loaded_config["Downsample_fs"]       == 5000
    assert loaded_config["Dynamic_Range_Lower"] == pytest.approx(-80.0)
    assert loaded_config["SA_Overlap"]          == pytest.approx(75.0)


def test_invalid_columns_raises(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame([{"wrong_col": "x", "another": "y"}]).to_csv(str(bad_path), index=False)
    with pytest.raises(ValueError):
        load_config(str(bad_path))


# export_config                                                        


def test_export_creates_file(loaded_config, tmp_path):
    out_path = str(tmp_path / "out.csv")
    export_config(out_path, loaded_config)
    assert os.path.isfile(out_path)


def test_export_has_correct_columns(loaded_config, tmp_path):
    out_path = str(tmp_path / "out.csv")
    export_config(out_path, loaded_config)
    df = pd.read_csv(out_path)
    assert "Config_Features" in df.columns
    assert "Config_Settings" in df.columns


def test_export_row_count(loaded_config, tmp_path):
    out_path = str(tmp_path / "out.csv")
    export_config(out_path, loaded_config)
    df = pd.read_csv(out_path)
    assert len(df) == len(CONFIG_KEYS)


# Round-trip                                                           


def test_round_trip(loaded_config, tmp_path):
    """
    Export a config then reload it. Every value should survive unchanged.
    This is the most important test — it proves load and export are
    inverse operations.
    """
    out_path   = str(tmp_path / "roundtrip.csv")
    export_config(out_path, loaded_config)
    reloaded   = load_config(out_path)

    for key in CONFIG_KEYS:
        original = loaded_config[key]
        restored = reloaded[key]
        if isinstance(original, float):
            assert restored == pytest.approx(original), f"Mismatch on {key}"
        else:
            assert restored == original, f"Mismatch on {key}"


# config_to_state_kwargs                                               


def test_state_kwargs_keys(loaded_config):
    kwargs = config_to_state_kwargs(loaded_config)
    expected_keys = [
        "file_chunk_number", "label_frame_length", "target_fs",
        "dynamic_range_l", "dynamic_range_u",
        "windowsize_cp", "overlap_cp", "min_f_cp", "max_f_cp",
        "windowsize_sa", "overlap_sa", "min_f_sa", "max_f_sa",
        "windowsize_sb", "overlap_sb", "min_f_sb", "max_f_sb",
        "windowsize_sc", "overlap_sc", "min_f_sc", "max_f_sc",
        "labels",
    ]
    for key in expected_keys:
        assert key in kwargs, f"Missing key: {key}"


def test_state_kwargs_labels_list(loaded_config):
    kwargs = config_to_state_kwargs(loaded_config)
    assert isinstance(kwargs["labels"], list)
    assert len(kwargs["labels"]) == 18


def test_state_kwargs_types(loaded_config):
    kwargs = config_to_state_kwargs(loaded_config)
    assert isinstance(kwargs["file_chunk_number"],  int)
    assert isinstance(kwargs["label_frame_length"], float)
    assert isinstance(kwargs["target_fs"],          int)
    assert isinstance(kwargs["windowsize_sa"],      int)