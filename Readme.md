# Generate SF WDL model based on data

Stockfish's "centipawn" evaluation is decoupled from the classical value
of a pawn, and is calibrated such that an advantage of
"100 centipawns" means the engine has a 50% probability to win
from this position in selfplay at fishtest LTC time control.

## Install
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

To update Stockfish's internal WDL model, the following steps are needed:

1. Obtain a large collection of engine-vs-engine games 
(at fishtest LTC time control) in pgn format and
save the pgn files in the `pgns` folder. This can, for example, be achieved
by running `python download_fishtest_pgns.py --path pgns` once a day.

2. Use `make` to compile `scoreWDLstat.cpp`, which will produce an executable
named `scoreWDLstat`.

3. Run `scoreWDLstat` to parse the pgn files in the `pgns` folder. A different
directory can be specified with `scoreWDLstat --dir <path-to-dir>`. The
computed WDL statistics will be stored in a file called `scoreWDLstat.json`.
The file will have entries of the form `"('D', 1, 78, 35)": 668132`, meaning
this tuple for `(outcome, move, material, eval)` was seen a total of 668132
times in the processed pgn files.

4. Run `python scoreWDL.py` to compute the WDL model parameters from the
data stored in `scoreWDLstat.json`. The script needs as input the value
`--NormalizeToPawnValue` from within Stockfish's
[`uci.h`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.h),
to be able to correctly convert the centipawn values from the pgn files to
the unit internally used by the engine. The script will output the new
values for `NormalizeToPawnValue` in
[`uci.h`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.h)
and `as[]`, `bs[]` in
[`uci.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp). See e.g. https://github.com/official-stockfish/Stockfish/pull/4373

## Results

<p align="center">
  <img src="WDL_model_summary.png?raw=true" width="1200">
</p>

## Contents

Other scripts that can be used to visualize different WDL data:

- `scoreWDLana_moves.py` : similar to `scoreWDL.py`, analyze wrt to moves
- `scoreWDLana_material.py` : similar to `scoreWDL.py`, analyze wrt to material
---
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
