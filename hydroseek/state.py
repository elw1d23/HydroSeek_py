from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class AppState:

    # Acoustic settings (Set Up tab)                                       
    file_chunk_number: int = 1
    label_frame_length: float = 5.0
    target_fs: int = 5000
    dynamic_range_l: float = -80.0
    dynamic_range_u: float = 10.0

    # Spectrogram settings                                                 
    
    windowsize_cp: int = 2048
    overlap_cp: float = 50.0
    min_f_cp: float = 10.0
    max_f_cp: float = 48000.0

    windowsize_sa: int = 2048
    overlap_sa: float = 75.0
    min_f_sa: float = 10.0
    max_f_sa: float = 48000.0
    mel_sa: bool = False

    windowsize_sb: int = 1024
    overlap_sb: float = 50.0
    min_f_sb: float = 10.0
    max_f_sb: float = 20000.0
    mel_sb: bool = False

    windowsize_sc: int = 512
    overlap_sc: float = 75.0
    min_f_sc: float = 10.0
    max_f_sc: float = 2000.0


    # Labels                                                               

    labels: list = field(default_factory=lambda: [f"NA_{i}" for i in range(1, 19)])

  
    # Runtime audio state (populated on Start)                            

    audio_data: Optional[np.ndarray] = None
    fs: int = 0
    audio_chunks: list = field(default_factory=list)
    audio_file_path: str = ""
    audio_length: float = 0.0          # total duration in seconds
    frame_size: int = 0                # samples per label frame
    num_frames_to_label: int = 0       # total frames across whole file
    num_chunks_per_split: int = 0      # frames per chunk (chunks 1 to N-1)
    frames_in_last_chunk: int = 0

    # Current navigation state
    current_chunk: Optional[np.ndarray] = None
    downsampled: Optional[np.ndarray] = None
    chunk: Optional[np.ndarray] = None       # current label frame (full Fs)
    ds_chunk: Optional[np.ndarray] = None    # current label frame (downsampled)
    current_plot_chunk: Optional[np.ndarray] = None  # context plot window

    start_index: int = 0
    end_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    down_start_index: int = 0
    down_end_index: int = 0

    # Counters 
    current_chunk_index: int = 0   # frame index within current big chunk
    current_frame_number: int = 0  # global frame counter across all chunks
    chunk_number: int = 0          # total frames in current big chunk
    num_chunks: int = 0            # alias kept for clarity

    # Computed overlap sample counts (derived from % settings)             #
    overlap_samples_sa: int = 0
    overlap_samples_sb: int = 0
    overlap_samples_sc: int = 0
    overlap_samples_cp: int = 0


    # Output                                                               
   
    labels_table: Optional[pd.DataFrame] = None

    # Event annotation state (point labels and bounding boxes)
    event_labels_table: Optional[pd.DataFrame] = None
    annotation_mode: str = "none"        # "none" | "point" | "box"
    current_event_label: str = ""        # free-text tag typed by the user
    event_id_counter: int = 0            # increments globally across the session

    # Display                                                              
    colormap: str = "turbo"   # matplotlib nearest-equivalent to parula

    def reset_navigation(self) -> None:
        """Call this when loading a new file to clear all runtime state."""
        self.audio_data = None
        self.audio_chunks = []
        self.current_chunk = None
        self.downsampled = None
        self.chunk = None
        self.ds_chunk = None
        self.current_plot_chunk = None
        self.start_index = 0
        self.end_index = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.down_start_index = 0
        self.down_end_index = 0
        self.current_chunk_index = 0
        self.current_frame_number = 0
        self.chunk_number = 0
        self.num_chunks = 0
        self.labels_table = None
        self.event_labels_table = None
        self.annotation_mode = "none"
        self.current_event_label = ""
        self.event_id_counter = 0