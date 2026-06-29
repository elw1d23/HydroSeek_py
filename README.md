# hydroseek-python
<img width="200" height="200" alt="logo" src="https://github.com/user-attachments/assets/3019b4ba-0f52-46ad-ade0-00103d5047e9" />

_Logo created with generative AI (openAI)_

# HydroSeek

A PyQt6 desktop application for labelling passive acoustic data
using spectrograms.
Allows for binary, multi-class labels with a counting feature, Bounding box annotation and point labels (v2.0)

Outputs a csv labelling table to the same location as the audio file labelled
Custom setup tab allows complete flexibiity over the labelling set up! Make sure you pay around with the optimal set up conditions for your data, and then export your config file for rapid set up each time. 

## Requirements

Python 3.11+

## Installation

## Installation

To use the code, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/elw1d23/HydroSeek_py.git
cd HydroSeek_py
```

2. Create and activate a virtual environment:

```bash
python -m venv hydro_env

# macOS/Linux
source hydro_env/bin/activate

# Windows
hydro_env\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```
