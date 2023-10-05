#!/bin/bash

# exit on errors
set -e

echo "started at: " `date`

# give range of commit SHAs to be considered for the WDL fitting
firstrev=70ba9de85cddc5460b1ec53e0a99bee271e26ece
lastrev=HEAD

# regex for book name
bookname=UHO_4060_v3.epd

# path for PGN files
pgnpath=pgns

# clone SF if needed
if [[ ! -e Stockfish ]]; then
    git clone https://github.com/official-stockfish/Stockfish.git >& clone.log
fi

# compile scoreWDLstat if needed
make >& make.log

# update SF and get a revision list
cd Stockfish
git checkout master >& checkout.log
git fetch origin >& fetch.log
git pull >& pull.log
revs=`git rev-list $firstrev^..$lastrev`

# get the currently valid value of NormalizeToPawnValue
oldpawn=`git grep 'const int NormalizeToPawnValue' $firstrev -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+'`
oldepoch=`git show --quiet --format=%ci $firstrev`
newepoch=`git show --quiet --format=%ci $lastrev`

# build a regex pattern to match all revisions
regex_pattern=""
for rev in $revs; do
    regex_pattern="${regex_pattern}.*-$rev|"
    newpawn=`git grep 'const int NormalizeToPawnValue' $rev -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+'`
    if [[ $oldpawn -ne $newpawn ]]; then
        echo "Revision $rev has wrong NormalizeToPawnValue ($newpawn != $oldpawn)"
        exit 1
    fi
done

# remove the trailing "|"
regex_pattern="${regex_pattern%|}"

cd ..

echo "Look recursively in directory $pgnpath for games from SPRT tests using"\
     "books matching \"$bookname\" for SF revisions between $firstrev (from"\
     "$oldepoch) and $lastrev (from $newepoch)."

# obtain the WDL data from games of SPRT tests of the SF revisions of interest
./scoreWDLstat --dir $pgnpath -r --matchEngine $regex_pattern --matchBook "$bookname" --fixFEN --SPRTonly -o updateWDL.json >& scoreWDLstat.log

# fit the new WDL model, keeping anchor at move 32
# we ignore the first 2 full moves out of book for fitting (11=8+1+2), and the first 9 for (contour) plotting (18=8+1+9)
python scoreWDL.py updateWDL.json --plot save --yDataTarget 32 --yDataMin 11 --yDataMax 120 --yPlotMin 18 --NormalizeToPawnValue $oldpawn >& scoreWDL.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=`awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; print sum; exit}' scoreWDL.log`
newpawn=`grep -oP 'const int NormalizeToPawnValue = \K\d+' scoreWDL.log`

if [[ $newpawn -ne $oldpawn ]]; then
    echo "Based on $poscount positions, NormalizeToPawnValue should change from $oldpawn to $newpawn."
else
    echo "Based on $poscount positions, NormalizeToPawnValue should stay at $oldpawn."
fi

echo "ended at: " `date`
