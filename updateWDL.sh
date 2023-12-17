#!/bin/bash

# exit on errors
set -e

# set range of commits to be considered for the WDL fitting, and other options
# by default we start from the most recent WDL model change and go to master
default_firstrev=70ba9de85cddc5460b1ec53e0a99bee271e26ece
default_lastrev=HEAD
default_moveMin=8
default_moveMax=120
firstrev=$default_firstrev
lastrev=$default_lastrev
moveMin=$default_moveMin
moveMax=$default_moveMax

while [[ $# -gt 0 ]]; do
    case "$1" in
    --firstrev)
        firstrev="$2"
        shift 2
        ;;
    --lastrev)
        lastrev="$2"
        shift 2
        ;;
    --moveMin)
        moveMin="$2"
        shift 2
        ;;
    --moveMax)
        moveMax="$2"
        shift 2
        ;;
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --firstrev FIRSTREV   First SF commit to collect games from (default: $default_firstrev)"
        echo "  --lastrev LASTREV     Last SF commit to collect games from (default: $default_lastrev)"
        echo "  --moveMin MOVEMIN     Parameter passed to scoreWDL.py (default: $default_moveMin)"
        echo "  --moveMax MOVEMAX     Parameter passed to scoreWDL.py (default: $default_moveMax)"
        exit 0
        ;;
    *)
        break
        ;;
    esac
done

echo "Running: $0 --firstrev=$firstrev --lasttrev=$lastrev --moveMin=$moveMin --moveMax=$moveMax"

echo "started at: " $(date)

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
python scoreWDL.py updateWDL.json --plot save --pgnName updateWDL.png --momType "move" --momTarget 32 --moveMin "$moveMin" --moveMax "$moveMax" --modelFitting optimizeProbability --NormalizeToPawnValue $oldpawn >&scoreWDL.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=$(awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; print sum; exit}' scoreWDL.log)
newpawn=$(grep -oP 'const int NormalizeToPawnValue = \K\d+' scoreWDL.log)

if [[ $newpawn -ne $oldpawn ]]; then
    echo "Based on $poscount positions, NormalizeToPawnValue should change from $oldpawn to $newpawn."
else
    echo "Based on $poscount positions, NormalizeToPawnValue should stay at $oldpawn."
fi

echo "ended at: " $(date)
