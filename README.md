# hydroseek-python
<img width="1254" height="1254" alt="logo" src="https://github.com/user-attachments/assets/3019b4ba-0f52-46ad-ade0-00103d5047e9" />

_Logo created with generative AI (openAI)_

# HydroSeek

A PyQt6 desktop application for labelling passive acoustic data
using spectrograms.
V1.0 allows for binary, multi-class labels with a counting feature.
Bounding boxes and point labels in development (v2.0)

Outputs a csv labelling table to the same location as the audio file labelled
Custom setup tab allows complete flexibiity over the labelling set up! Make sure you pay around with the optimal set up conditions for your data, and then export your config file for rapid set up each time. 

## Requirements

Python 3.11+

## Installation

To use the code, follow these steps:

Clone the repository:
'git clone https://github.com/elw1d23/HydroSeek_py.git'

cd HydroSeek_py

Create a virtual environment and activate it:
'python -m venv hydro_env'
'source hydro_env/bin/activate' # On Windows use `hydro_env\Scripts\activate`

Install the required packages:
'pip install -r requirements.txt'

## Running

python main.py
