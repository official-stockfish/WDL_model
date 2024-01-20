import argparse, datetime, json, os, re, tarfile, urllib.request

parser = argparse.ArgumentParser(
    description="Bulk-download .pgn.gz files from finished tests on fishtest.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./pgns",
    help="Downloaded .pgn.gz files will be stored in PATH/date/test-id/.",
)
parser.add_argument(
    "--time_delta",
    type=float,
    default=168.0,  # 1 week
    help="Delta of hours from now since the desired tests are last updated.",
)
parser.add_argument(
    "--ltc_only",
    type=bool,
    default=True,
    help="A flag for LTC tests only.",
)
parser.add_argument(
    "--success_only",
    type=bool,
    default=False,
    help="A flag for Green tests only.",
)
parser.add_argument(
    "--yellow_only",
    type=bool,
    default=False,
    help="A flag for yellow tests only.",
)
parser.add_argument(
    "--username",
    type=str,
    default="",
    help="Specified username to download from their finished tests.",
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

# Get the current UTC time
current_time_utc = datetime.datetime.now(datetime.timezone.utc)

# Subtract hours from the current time
time_difference = datetime.timedelta(hours=args.time_delta)
result_time_utc = current_time_utc - time_difference

# Convert the result to a Unix timestamp
unix_timestamp = result_time_utc.timestamp()

additional_query_params = f"&timestamp={unix_timestamp}"
if args.ltc_only:
    additional_query_params += "&ltc_only=1"
if args.success_only:
    additional_query_params += "&success_only=1"
if args.yellow_only:
    additional_query_params += "&yellow_only=1"
if args.username != "":
    additional_query_params += f"&username={args.username}"

page = 1

while True:
    # fetch from desired page of finished tests, parse and list new IDs
    url = f"https://tests.stockfishchess.org/api/finished_runs?page={page}{additional_query_params}"
    try:
        with urllib.request.urlopen(url) as response:
            response_data = response.read().decode("utf-8")
            response_json = json.loads(response_data)
            if response_json is None or not response_json:
                break

            ids = [
                (
                    id,
                    datetime.datetime.strptime(
                        response_json[id]["start_time"], "%Y-%m-%d %H:%M:%S.%f"
                    ).strftime("%y-%m-%d"),
                    response_json[id],
                )
                for id in response_json
                if not id in downloaded
            ]
    except urllib.error.HTTPError as ex:
        print(f"HTTP Error: {ex.code} - {ex.reason}")
    except urllib.error.URLError as ex:
        print(f"URL Error: {ex.reason}")
    except json.JSONDecodeError as ex:
        print(f"JSON Decoding Error: {ex}")
    except Exception as ex:
        print(f"Error: {ex}")

    # download .pgn.tar ball for each test
    for test, dateStr, meta in ids:
        path = args.path + dateStr + "/" + test + "/"
        if not os.path.exists(args.path + dateStr):
            os.makedirs(args.path + dateStr)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.verbose >= 1:
            print(f"Collecting meta data for test {test} ...")
        wins, draws, losses = 0, 0, 0
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
    page += 1  # Move to the next page for the next iteration
print("Finished downloading PGNs.")
