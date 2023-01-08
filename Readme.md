# Generate SF WLD model based on data

## Usage

To generate the WLD model two steps are needed:

1. Run scoreWLDstat.py to parse a large collection of pgn files providing the data. By default the script will look for pgn files in the pgn folder that end with `.pgn`, unless a different directory is specified. 

2. Run scoreWLDana_moves_fit.py to compute the model parameters
For this step, specify the with option `--NormalizePawnValue` the correct number to convert pawn scores (as assumed to be in the pgn) to the units internally used by the engine.


## Results

See e.g. https://github.com/official-stockfish/Stockfish/pull/3981 (older version) also available as images in examples/

<p align="center">
  <img src="WLD_model_summary.png?raw=true" width="1200">
</p>

## Contents

further tools can be used to experiment:

* scoreWLDstat.py  : extract counts of game outcomes as a function of score, material count and move number (needs fishtest games)
                     produces scoreWLDstat.json : extracted data from games
                     "('D', 1, 78, 35)": 668132, -> outcome = draw, move = 1, material = 78, eval = 35 cp -> 668132 positions
*  scoreWLDana_moves_fit.py : fit some models

*  scoreWLDana_moves.py : similar to above, analyze wrt to moves
*  scoreWLDana_material.py : similar to above analyze wrt to material


