import argparse, json, os, re, requests, tarfile, urllib.request

parser = argparse.ArgumentParser(
    description="Bulk-download .pgn.gz files from completed LTC tests on fishtest.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./pgns",
    help="Downloaded .pgn.gz files will be stored in PATH/date/test-id/.",
)
parser.add_argument(
    "--page",
    type=int,
    default=1,
    help="Page of completed LTC tests to download from.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase output with e.g. -v or -vv.",
)
args = parser.parse_args()
if args.path == "":
    args.path = "./"
elif args.path[-1] != "/":
    args.path += "/"

if not os.path.exists(args.path):
    os.makedirs(args.path)

# find the set of fully downloaded Ids (looking in the full file tree)
p = re.compile("([a-z0-9]*)-[0-9]*.pgn(|.gz)")  # match any testId-runId.pgn(.gz)
downloaded = set()

for _, _, files in os.walk(args.path):
    for name in files:
        m = p.match(name)
        if m:
            downloaded.add(m.group(1))

print(f"Found {len(downloaded)} downloaded tests in {args.path} already.")

# fetch from desired page of finished LTC tests, parse and list new IDs
url = f"https://tests.stockfishchess.org/tests/finished?ltc_only=1&page={args.page}"
p = re.compile('<a href="/tests/view/([a-z0-9]*)">')

ids = []
response = urllib.request.urlopen(url)
webContent = response.read().decode("utf-8").splitlines()
nextlineDate, dateStr = False, ""
for line in webContent:
    m = p.search(line)
    if m:
        testId = m.group(1)
        if not testId in downloaded:
            ids.append((testId, dateStr))
    else:
        if nextlineDate:
            dateStr = line.strip()
        nextlineDate = line.endswith('"run-date">')

# download metadata and .pgn.tar ball for each test
for test, dateStr in ids:
    path = args.path + dateStr + "/" + test + "/"
    if not os.path.exists(args.path + dateStr):
        os.makedirs(args.path + dateStr)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.verbose >= 1:
        print(f"Collecting meta data for test {test} ...")
    url = "https://tests.stockfishchess.org/api/get_run/" + test
    wins, draws, losses = 0, 0, 0
    try:
        meta = requests.get(url, timeout=30)
        meta.raise_for_status()
        meta = meta.json()
        if "spsa" in meta.get("args", {}):
            if args.verbose >= 1:
                print(f"Skipping SPSA test {test} ...")
            continue
        with open(path + test + ".json", "w") as jsonFile:
            json.dump(meta, jsonFile, indent=4, sort_keys=True)
        if "results" in meta:
            wins = meta["results"].get("wins", 0)
            draws = meta["results"].get("draws", 0)
            losses = meta["results"].get("losses", 0)
    except Exception as ex:
        if args.verbose >= 2:
            print(f'  error: {test}.json : caught exception "{ex}"')

    url = "https://tests.stockfishchess.org/api/run_pgns/" + test + ".pgns.tar"
    try:
        response = urllib.request.urlopen(url)
        mb = int(response.getheader("Content-Length", 0)) // (2**20)
        games = wins + draws + losses
        msg = f"Downloading{'' if mb == 0 else f' {mb}MB'} .pgns.tar file "
        if games:
            msg += f"with {games} games {'' if args.verbose == 0 else f'(WDL = {wins} {draws} {losses}) '}"
        print(msg + f"to {path} ...")
        tmpName = path + test + ".tmp"
        urllib.request.urlretrieve(url, tmpName)
        if args.verbose >= 2:
            print("Extracting the tar file ...")
        with tarfile.open(tmpName, "r") as tar:
            tar.extractall(path)
        os.remove(tmpName)
    except Exception as ex:
        if args.verbose >= 2:
            print(f'  error: caught exception "{ex}"')
        continue
