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

## Background

The underlying assumption of WDL model is that the win rate for a position
can be well modeled as a function of the evaluation of that position.
The data shows a [logistic function](https://en.wikipedia.org/wiki/Logistic_function) (see also logistic regression)
gives a good approximation of the win rate as a function of the evaluation.

```
win_rate(x) = 1 / ( 1 + exp(-(x-a)/b))
```
In this equation, the parameters a and b need to be fitted to the data,
which is the purpose of this repository. a is the evaluation for which a 50% win rate is observed,
While b indicates how quickly this rate changes with evaluation. A small b indicates that small
changes in eval quickly turn a 'game on the edge' (i.e. 50% win rate) into a dead draw or a near certain win.

The model furthermore assumes symmetry in evaluation, so that the following quantities follow as well:
```
loss_rate(x) = win_rate(-x)
draw_rate(x) = 1 - win_rate(x) - loss_rate(x)
```

This information also allows for estimating the game score
```
score(x) = 1 * win_rate(x) + 0.5 * draw_rate(x) + 0 * loss_rate(x)
```

The model is made more accurate by not only taking the evaluation,
but also the material or game move counter (mom) into account. This dependency is modeled by making the
parameters a and b a function of mom. The win/draw/loss rates are now 2D functions, while a and b are 1D functions.
For simplicity the 1D functions a and b are represented as a polynomial of 3rd degree.

The parameters that need to be fit to represent the model completely are thus the 8 parameters that
determine these two polynomials. For example:
```
a(x) = ((-1.719 * x / 32 + 12.448) * x / 32 + -12.855) * x / 32 + 331.883
b(x) = ((-3.001 * x / 32 + 22.505) * x / 32 + -51.253) * x / 32 + 93.209
```

Two fit these parameters various approaches exist, ranging from a simple fit of the observed win rate,
to a somewhat more elaborate maximization of the probability of predicting the correct game outcome
for the available data.

## Install

Python 3.9 or higher is required.

```
pip install -r requirements.txt
```

C++17 compatible compiler is required and `zlib` needs to be present.
```
sudo apt-get install zlib1g-dev
````

## Usage
_To allow for efficient analysis multiple pgn files are analysed in parallel.
Analysis of a single pgn file is not parallelized. Files can either be in `.pgn`
or `.pgn.gz` format. The script will automatically detect the file format and
decompress `.pgn.gz` files on the fly._

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
