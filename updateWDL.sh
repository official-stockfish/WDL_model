#!/bin/bash

# exit on errors
set -e

echo "started at: " $(date)

# give range of commit SHAs to be considered for the WDL fitting
firstrev=70ba9de85cddc5460b1ec53e0a99bee271e26ece
lastrev=HEAD

# regex for book name
bookname="UHO_4060_v..epd|UHO_Lichess_4852_v1.epd"

# path for PGN files
pgnpath=pgns

outpath=update
logdir=logs
logpath="$outpath"/"$logdir"

# create log directory if needed
mkdir -p "$logpath"

# clone repos if needed, and pull latest revisions
for repo in "Stockfish" "books"; do
    if [[ ! -e "$outpath"/"$repo" ]]; then
        git clone https://github.com/official-stockfish/"$repo".git ./"$outpath"/"$repo" >& ./"$logdir"/clone.log
    fi
    cd ./"$outpath"/"$repo"
    echo $(pwd)
    git checkout master >& ../"$logdir"/checkout-"$repo".log
    git fetch origin >& ../"$logdir"/fetch-"$repo".log
    git pull >& ../"$logdir"/pull-"$repo".log
    cd ../..
done

# create a sorted list of all the books matching the regex
matching_books=()
for file in $(find "$outpath"/books -type f -name "*.zip" | sort); do
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
fixfen="$outpath"/"fixfen_$bookhash.epd"

if [[ ! -e "$fixfen.gz" ]]; then
    rm -f "$fixfen"
    for book in "${matching_books[@]}"; do
        unzip -o "$outpath"/books/"$book".zip >& ./"$logpath"/unzip"$book".log
        awk 'NF >= 6' "$book" >>"$fixfen"
        rm "$book"
    done
    sort -u "$fixfen" -o _tmp_"$fixfen" && mv _tmp_"$fixfen" "$fixfen"
    gzip "$fixfen"
fi

# get a SF revision list
cd "$outpath"/Stockfish
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

cd ../..

# compile scoreWDLstat if needed
make >& ./"$logpath"/make.log

echo "Look recursively in directory $pgnpath for games from SPRT tests using" \
    "books matching \"$bookname\" for SF revisions between $firstrev (from" \
    "$oldepoch) and $lastrev (from $newepoch)."

# obtain the WDL data from games of SPRT tests of the SF revisions of interest
./scoreWDLstat --dir $pgnpath -r --matchRev $regex_pattern --matchBook "$bookname" --fixFENsource "$fixfen.gz" --SPRTonly -o ./"$outpath"/updateWDL.json >& ./"$logpath"/scoreWDLstat.log

# fit the new WDL model, keeping anchor at move 32
# we ignore the first 2 full moves out of book for fitting (11=8+1+2), and the first 9 for (contour) plotting (18=8+1+9)
python scoreWDL.py ./"$outpath"/updateWDL.json --plot save --pgnName ./"$outpath"/updateWDL.png --yDataTarget 32 --yDataMin 8 --yDataMax 120 --yPlotMin 8 --modelFitting optimizeProbability --NormalizeToPawnValue $oldpawn >& ./"$logpath"/scoreWDL.log

# extract the total number of positions, and the new NormalizeToPawnValue
poscount=$(awk -F '[() ,]' '/Retained \(W,D,L\)/ {sum = 0; for (i = 9; i <= NF; i++) sum += $i; print sum; exit}' ./$logpath/scoreWDL.log)
newpawn=$(grep -oP 'const int NormalizeToPawnValue = \K\d+' ./$logpath/scoreWDL.log)

if [[ $newpawn -ne $oldpawn ]]; then
    echo "Based on $poscount positions, NormalizeToPawnValue should change from $oldpawn to $newpawn."
else
    echo "Based on $poscount positions, NormalizeToPawnValue should stay at $oldpawn."
fi

echo "ended at: " $(date)
