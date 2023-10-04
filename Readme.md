# Generate SF WDL model based on data

Stockfish's "centipawn" evaluation is decoupled from the classical value
of a pawn, and is calibrated such that an advantage of
"100 centipawns" means the engine has a 50% probability to win
from this position in selfplay at move 32 at fishtest LTC time control.\
If the option `UCI_ShowWDL` is enabled, the engine will show Win-Draw-Loss
probabilities alongside its "centipawn" evaluation. These probabilities
depend on the engine's evaluation and the move number, and are computed
from a WDL model that can be generated from fishtest data with the help of
the scripts in this repository.

## Install

Python 3.9 or higher is required.

```
pip install -r requirements.txt
```

## Usage

To update Stockfish's internal WDL model, the following steps are needed:

1. Obtain a large collection of engine-vs-engine games (at fishtest LTC 
time control) by regularly running `python download_fishtest_pgns.py` 
over a period of time. The script will download the necessary pgn files
and metadata describing the test conditions from 
[fishtest](https://tests.stockfishchess.org/).

2. Run the script `updateWDL.sh`, which will automatically perform these
steps:

    - Run `make` to compile `scoreWDLstat.cpp`, producing the executable 
      `scoreWDLstat`.

    - Run `scoreWDLstat` with some custom parameters to parse the downloaded
      pgn files. The computed WDL statistics will be stored in a file called 
      `updateWDL.json`. The file will have entries of the form 
      `"('D', 1, 78, 35)": 668132`, meaning this tuple for 
      `(outcome, move, material, eval)` was seen a total of 668132 times in 
      the processed pgn files.

    - Run `python scoreWDL.py` with some custom parameters to compute the WDL 
      model parameters from the data stored in `updateWDL.json`. The script's
      output will be stored in `scoreWDL.log` and will contain the new
      values for `NormalizeToPawnValue` in Stockfish's
      [`uci.h`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.h)
      and `as[]`, `bs[]` in
      [`uci.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp). See e.g. https://github.com/official-stockfish/Stockfish/pull/4373.
      In addition, the script will produce a graphical illustration of the 
      analysed data and the fitted WDL model in the file
      `WDL_model_summary.png`, as displayed below.

## Results

<p align="center">
  <img src="WDL_model_summary.png?raw=true" width="1200">
</p>

## Help and other options

Running `scoreWDLstat --help` and `python scoreWDL.py --help`, respectively,
will provide a description of possible command line options for the two
programs. For example:

- `scoreWDLstat --matchEngine <regex>` : extracts WDL data only from the
   engine matching the regex
- `python scoreWDL.py --yDataTarget 30` : chooses move 30 (rather than 32)
  as target move for the 100cp anchor
- `python scoreWDL.py --yData material --yDataTarget 68` : bases the fitting
  on material (rather than move), with 100cp anchor a material count of 68
---
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
