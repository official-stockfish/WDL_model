# Generate SF WLD model based on data

## Usage

To generate the WLD model three steps are needed:

1. You first need to compile `scoreWLDstat.cpp`. The simplest way to do that is to use the supplied Makefile, in which case you only need to run `make` in the current directory. This will output an executable named `main`.

2. Run `main` to parse a large collection of pgn files providing the data. By default the script will look for pgn files in the `pgns` folder that end with `.pgn`, unless a different directory is specified using `main --dir \<path-to-dir\>`.

3. Run scoreWLDana_moves_fit.py to compute the model parameters
   For this step, specify the with option `--NormalizeToPawnValue` the correct number to convert pawn scores (as assumed to be in the pgn) to the units internally used by the engine.

## Results

<p align="center">
  <img src="WLD_model_summary.png?raw=true" width="1200">
</p>

See e.g. https://github.com/official-stockfish/Stockfish/pull/4373

## Contents

further tools can be used to experiment:

- scoreWLDstat.cpp : extract counts of game outcomes as a function of score, material count and move number (needs fishtest games)
  produces scoreWLDstat.json : extracted data from games
  "('D', 1, 78, 35)": 668132, -> outcome = draw, move = 1, material = 78, eval = 35 cp -> 668132 positions
- scoreWLDana_moves_fit.py : fit some models

- scoreWLDana_moves.py : similar to above, analyze wrt to moves
- scoreWLDana_material.py : similar to above analyze wrt to material
