# Generate SF WLD model based on data

## Usage

To generate the WLD model two steps are needed:

1. Run scoreWLDstat.py to parse a large collection of pgn files providing the data. By default the script will look for pgn files in the pgn folder that end with `.pgn`, unless a different directory is specified. 

2. Run scoreWLDana_moves_fit.py to compute the model parameters
For this step, specify the with option `--NormalizePawnValue` the correct number to convert pawn scores (as assumed to be in the pgn) to the units internally used by the engine.


## Results

See e.g. https://github.com/official-stockfish/Stockfish/pull/3981 also available as images in examples/

### Sample win rate model at move 30
<p align="center">
  <img src="examples/Figure_1.png?raw=true" width="1200">
</p>
### a,b parameter dependence with move number
<p align="center">
  <img src="examples/Figure_2.png?raw=true" width="1200">
</p>
### 2D contour plot of the data
<p align="center">
  <img src="examples/Figure_3.png?raw=true" width="1200">
</p>
### 2D contour plot of the model
<p align="center">
  <img src="examples/Figure_4.png?raw=true" width="1200">
</p>

## Contents

further tools can be used to experiment:

* scoreWLDstat.py  : extract counts of game outcomes as a function of score, material count and move number (needs fishtest games)
                     produces scoreWLDstat.json : extracted data from games
                     "('D', 1, 78, 35)": 668132, -> outcome = draw, move = 1, material = 78, eval = 35 cp -> 668132 positions
*  scoreWLDana_moves_fit.py : fit some models

*  scoreWLDana_moves.py : similar to above, analyze wrt to moves
*  scoreWLDana_material.py : similar to above analyze wrt to material


