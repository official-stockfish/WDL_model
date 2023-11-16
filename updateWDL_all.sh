#!/bin/bash

# exit on errors
set -e

echo "started at: " $(date)

# give range of commit SHAs to be considered for the WDL fitting
firstrev=68e1e9b3811e16cad014b590d7443b9063b3eb52
lastrev=HEAD
lastrev=b25d68f6ee2d016cc0c14b076e79e6c44fdaea2a

# regex for book name
bookname="UHO_4060_v..epd|UHO_Lichess_4852_v1.epd"

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

# create a sorted list of all the books matching the regex
matching_books=()
for file in $(find books -type f -name "*.zip" | sort); do
    book=$(basename "$file" .zip)
    if [[ $book =~ $bookname ]]; then
        matching_books+=("$book")
    fi
done

if [ ${#matching_books[@]} -eq 0 ]; then
    echo "No matching books found for the regex $bookname."
    exit 1
fi

# refetch books if the list of matching books is new
bookhash=$(echo -n "${matching_books[@]}" | md5sum | cut -d ' ' -f 1)
fixfen="fixfen_$bookhash.epd"

if [[ ! -e "$fixfen.gz" ]]; then
    rm -f "$fixfen"
    for book in "${matching_books[@]}"; do
        unzip -o books/"$book".zip >&unzip.log
        awk 'NF >= 6' "$book" >>"$fixfen"
        rm "$book"
    done
    sort -u "$fixfen" -o _tmp_"$fixfen" && mv _tmp_"$fixfen" "$fixfen"
    gzip "$fixfen"
fi

# get a SF revision list
cd Stockfish
revs=$(git rev-list $firstrev^..$lastrev)
cd ..

# compile scoreWDLstat if needed
make >&make.log

for rev in $revs; do

# get the currently valid value of NormalizeToPawnValue
cd Stockfish
oldpawn=$(git grep 'const int NormalizeToPawnValue' $rev -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+')
revdate=$(git show -s --format=%cd --date=format:'%Y-%m-%d' $rev)
cd ..

regex_pattern=".*${rev:0:10}.*"
./scoreWDLstat --dir $pgnpath -r --matchRev $regex_pattern --matchBook "$bookname" --fixFENsource "$fixfen.gz" --SPRTonly -o updateWDL_$rev.json >&scoreWDLstat_$rev.log

python scoreWDL.py updateWDL_$rev.json --plot save --pgnName updateWDL_$rev.png --yDataTarget 32 --yDataMin 8 --yDataMax 120 --yPlotMin 8 --modelFitting optimizeProbability --NormalizeToPawnValue $oldpawn >&scoreWDL_$rev.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=$(awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; print sum; exit}' scoreWDL_$rev.log)
newpawn=$(grep -oP 'const int NormalizeToPawnValue = \K\d+' scoreWDL_$rev.log || true)

echo "$rev $revdate $poscount $newpawn" | tee -a full_ana.out

done

echo "ended at: " $(date)
