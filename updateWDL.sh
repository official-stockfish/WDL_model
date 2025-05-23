#!/bin/bash

# exit on errors
set -e

# set range of commits to be considered for the WDL fitting, and other options
# by default we start from the most recent WDL model change and go to master
default_firstrev=4a869f41c6113f1ccdd0f11551858fdc849a245a
default_lastrev=HEAD
default_materialMin=17
default_EloDiffMax=5
firstrev=$default_firstrev
lastrev=$default_lastrev
materialMin=$default_materialMin
EloDiffMax=$default_EloDiffMax

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
    --materialMin)
        materialMin="$2"
        shift 2
        ;;
    --EloDiffMax)
        EloDiffMax="$2"
        shift 2
        ;;
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --firstrev FIRSTREV        First SF commit to collect games from (default: $default_firstrev)"
        echo "  --lastrev LASTREV          Last SF commit to collect games from (default: $default_lastrev)"
        echo "  --materialMin MATERIALMIN  Parameter passed to scoreWDL.py (default: $default_materialMin)"
        echo "  --EloDiffMax ELODIFFMAX    Parameter passed to scoreWDLstat (default: $default_EloDiffMax)"
        exit 0
        ;;
    *)
        break
        ;;
    esac
done

echo "Running: $0 --firstrev $firstrev --lasttrev $lastrev --materialMin $materialMin --EloDiffMax $EloDiffMax"

echo "started at: " $(date)

# regex for book name
bookname="UHO_Lichess_4852_v..epd"

# path for PGN files
pgnpath=pgns

# clone repo(s) if needed, and pull latest revisions
for repo in "Stockfish"; do
    if [[ ! -e "$repo" ]]; then
        git clone https://github.com/official-stockfish/"$repo".git >&clone.log
    fi
    cd "$repo"
    git checkout master >&checkout.log
    git fetch origin >&fetch.log
    git pull >&pull.log
    cd ..
done

# get a SF revision list
cd Stockfish
revs=$(git rev-list $firstrev^..$lastrev)

get_pawn_value() {
    # extract pawn value from "--NormalizeToPawnValue" or "--NormalizeData" string
    local pawn=$(echo "$1" | awk '/--NormalizeToPawnValue/ {print $2}')
    if [ -z "$pawn" ]; then
        # return the rounded sum of the coefficients in as
        pawn=$(echo "$1" | grep -oP '"as":\[\K[^\]]+' | tr ',' '\n' | awk '{s+=$1} END {printf "%.0f", s}')
    fi
    echo "$pawn"
}
get_normalize_data() {
    # construct the "--NormalizeToPawnValue" or "--NormalizeData" string
    local revision="$1"
    local pawn=$(git grep 'const int NormalizeToPawnValue' "$revision" -- src/uci.h | grep -oP 'const int NormalizeToPawnValue = \K\d+')

    if [ -z "$pawn" ]; then
        pawn=$(git grep 'constexpr int  NormalizeToPawnValue' "$revision" -- src/uci.cpp | grep -oP 'constexpr int  NormalizeToPawnValue = \K\d+')
    fi

    if [ -z "$pawn" ]; then
        line=$(git grep 'double m = std::clamp(material' "$revision" -- src/uci.cpp)

        momMin="${line#*std::clamp(material, }"
        momMin="${momMin%%,*}"
        momMax="${line##*, }"
        momMax="${momMax%%)*}"
        momTarget="${line##* }"
        momTarget="${momTarget%.0*}"

        line=$(git grep 'constexpr double as\[\] = {' "$revision" -- src/uci.cpp | grep -oP 'constexpr double as\[\] = {.*')
        as="${line#*constexpr double as[] = \{}"
        as="${as%\};}"
        as=$(sed 's/ //g' <<<"$as") # remove spaces

        echo "--NormalizeData {\"momType\":\"material\",\"momMin\":$((momMin)),\"momMax\":$((momMax)),\"momTarget\":$((momTarget)),\"as\":[$as]}"
    else
        echo "--NormalizeToPawnValue $pawn"
    fi
}

# get the currently valid value of NormalizeData
oldnormdata=$(get_normalize_data "$firstrev")
oldpawn=$(get_pawn_value "$oldnormdata")
oldepoch=$(git show --quiet --format=%ci $firstrev)
newepoch=$(git show --quiet --format=%ci $lastrev)

# build a regex pattern to match all revisions
regex_pattern=""
for rev in $revs; do
    regex_pattern="${regex_pattern}.*$rev|"
    newnormdata=$(get_normalize_data "$rev")
    if [[ "$oldnormdata" != "$newnormdata" ]]; then
        echo "Revision $rev has wrong NormalizeData ($newnormdata != $oldnormdata)"
        exit 1
    fi
done

# remove the trailing "|"
regex_pattern="${regex_pattern%|}"

cd ..

# compile scoreWDLstat if needed
make >&make.log

echo "Look recursively in directory $pgnpath for games with max nElo" \
    "difference $EloDiffMax using" \
    "books matching \"$bookname\" for SF revisions between $firstrev (from" \
    "$oldepoch) and $lastrev (from $newepoch)."

# obtain the WDL data from games of the SF revisions of interest
./scoreWDLstat --dir $pgnpath -r --matchTC "60\+0.6" --matchThreads 1 --EloDiffMax $EloDiffMax --matchRev $regex_pattern --matchBook "$bookname" -o updateWDL.json >&scoreWDLstat.log

gamescount=$(grep -o '[0-9]\+ games' scoreWDLstat.log | grep -o '[0-9]\+')

if [[ $gamescount -eq 0 ]]; then
    echo "No games found for revisions of interest."
    exit 0
fi

# fit the new WDL model, keeping anchor at material 58
python scoreWDL.py updateWDL.json --plot save --pngName updateWDL.png --pngNameDistro updateWDLdistro.png --momType material --momTarget 58 --materialMin $materialMin --moveMin 1 --modelFitting optimizeProbability $oldnormdata >&scoreWDL.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=$(awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; printf "%.0f\n", sum; exit}' scoreWDL.log)

if [[ $poscount -eq 0 ]]; then
    echo "No positions found."
    exit 0
fi

newpawn=$(grep -oP 'const int NormalizeToPawnValue = \K\d+' scoreWDL.log)

if [[ $newpawn -ne $oldpawn ]]; then
    echo "Based on $poscount positions from $gamescount games, NormalizeToPawnValue should change from $oldpawn to $newpawn."
else
    echo "Based on $poscount positions from $gamescount games, NormalizeToPawnValue should stay at $oldpawn."
fi

echo "ended at: " $(date)
