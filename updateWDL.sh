#!/bin/bash

# exit on errors
set -e

echo "started at: " $(date)

# give range of commit SHAs to be considered for the WDL fitting
firstrev=70ba9de85cddc5460b1ec53e0a99bee271e26ece
lastrev=HEAD

# regex for book name
bookname="UHO_4060_v..epd|UHO_Lichess_4852_v1.epd"
bookname="UHO_Lichess_4852_v1.epd"

# path for PGN files
pgnpath=pgns

# clone repos if needed, and pull latest revisions
for repo in "Stockfish" "books"; do
    if [[ ! -e "$repo" ]]; then
        git clone https://github.com/official-stockfish/"$repo".git >&clone.log
    fi
    cd "$repo"
    git checkout master >&checkout.log
    git fetch origin >&fetch.log
    git pull >&pull.log
    cd ..
done

# get the books necessary for the move counter fixing
bookhash=$(echo -n "$bookname" | md5sum | cut -d ' ' -f 1)
fixfen=fixfen_"$bookhash".epd
if [[ ! -e "$fixfen.gz" ]]; then
    rm -f "$fixfen"
    for file in books/*.zip; do
        book=$(basename "$file" .zip)
        if [[ $book =~ $bookname ]]; then
            unzip -o "$file" >&unzip.log
            awk 'NF >= 6' "$book" >>"$fixfen"
            rm "$book"
        fi
    done
    sort -u "$fixfen" -o _tmp_"$fixfen" && mv _tmp_"$fixfen" "$fixfen"
    gzip "$fixfen"
fi

# get a SF revision list
cd Stockfish
revs=$(git rev-list $firstrev^..$lastrev)

# get the currently valid value of NormalizeToPawnValue
oldpawn=$(git grep 'const int NormalizeToPawnValue' $firstrev -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+')
oldepoch=$(git show --quiet --format=%ci $firstrev)
newepoch=$(git show --quiet --format=%ci $lastrev)

# build a regex pattern to match all revisions
regex_pattern=""
for rev in $revs; do
    regex_pattern="${regex_pattern}.*$rev|"
    newpawn=$(git grep 'const int NormalizeToPawnValue' $rev -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+')
    if [[ $oldpawn -ne $newpawn ]]; then
        echo "Revision $rev has wrong NormalizeToPawnValue ($newpawn != $oldpawn)"
        exit 1
    fi
done

# remove the trailing "|"
regex_pattern="${regex_pattern%|}"

cd ..

# compile scoreWDLstat if needed
make >&make.log

echo "Look recursively in directory $pgnpath for games from SPRT tests using" \
    "books matching \"$bookname\" for SF revisions between $firstrev (from" \
    "$oldepoch) and $lastrev (from $newepoch)."

# obtain the WDL data from games of SPRT tests of the SF revisions of interest
./scoreWDLstat --dir $pgnpath -r --matchRev $regex_pattern --matchBook "$bookname" --fixFENsource "$fixfen.gz" --SPRTonly -o updateWDL.json >&scoreWDLstat.log

# fit the new WDL model, keeping anchor at move 32
# we ignore the first 2 full moves out of book for fitting (11=8+1+2), and the first 9 for (contour) plotting (18=8+1+9)
python scoreWDL.py updateWDL.json --plot save --yDataTarget 32 --yDataMin 11 --yDataMax 120 --yPlotMin 18 --NormalizeToPawnValue $oldpawn >&scoreWDL.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=$(awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; print sum; exit}' scoreWDL.log)
newpawn=$(grep -oP 'const int NormalizeToPawnValue = \K\d+' scoreWDL.log)

if [[ $newpawn -ne $oldpawn ]]; then
    echo "Based on $poscount positions, NormalizeToPawnValue should change from $oldpawn to $newpawn."
else
    echo "Based on $poscount positions, NormalizeToPawnValue should stay at $oldpawn."
fi

echo "ended at: " $(date)
