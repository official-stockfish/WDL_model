import argparse, datetime, json, gzip, os, re, urllib.request


def format_large_number(number):
    suffixes = ["", "K", "M", "G", "T", "P"]
    for suffix in suffixes:
        if number < 1000:
            return f"{number:.0f}{suffix}"
        number /= 1000
    return f"{number:.0f}{suffixes[-1]}"


def open_file_rt(filename):
    # allow reading text files either plain or in gzip format
    open_func = gzip.open if filename.endswith(".gz") else open
    return open_func(filename, "rt")


def count_games(filename):
    count = 0
    with open_file_rt(filename) as f:
        for line in f:
            if "Result" in line:
                if "1-0" in line or "0-1" in line or "1/2-1/2" in line:
                    count += 1
    return count


parser = argparse.ArgumentParser(
    description="Bulk-download .pgn.gz files from finished tests on fishtest.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./pgns",
    help="Downloaded .pgn.gz files will be stored in PATH/YY-MM-DD/test-Id/.",
)
parser.add_argument(
    "--time_delta",
    type=float,
    default=168.0,  # 1 week
    help="Delta of hours from now since the desired tests are last updated.",
)
parser.add_argument(
    "--ltc_only",
    type=str,
    default="True",
    help="A True/False flag for LTC tests only.",
)
parser.add_argument(
    "--tc_lower_limit",
    type=float,
    help="Download only tests where base tc for each side is at least this.",
)
parser.add_argument(
    "--tc_upper_limit",
    type=float,
    help="Download only tests where base tc for each side is at most this.",
)
parser.add_argument(
    "--success_only",
    type=str,
    default="False",
    help="A True/False flag for green tests only.",
)
parser.add_argument(
    "--yellow_only",
    type=str,
    default="False",
    help="A True/False flag for yellow tests only.",
)
parser.add_argument(
    "--username",
    type=str,
    help="Download USERNAME's tests only.",
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
# match any filename of the form testId-runId.pgn(.gz) or testId.pgn(.gz)
p = re.compile("([a-z0-9]*)(-[0-9]*)?\.pgn(|\.gz)")
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
if args.ltc_only.lower() == "true":
    additional_query_params += "&ltc_only=1"
if args.success_only.lower() == "true":
    additional_query_params += "&success_only=1"
if args.yellow_only.lower() == "true":
    additional_query_params += "&yellow_only=1"
if args.username is not None:
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
                    datetime.datetime.fromisoformat(
                        response_json[id]["start_time"]
                    ).strftime("%y-%m-%d"),
                    response_json[id],
                )
                for id in response_json
                if not id in downloaded
            ]
    except urllib.error.HTTPError as ex:
        print(f"HTTP Error: {ex.code} - {ex.reason}")
        break
    except urllib.error.URLError as ex:
        print(f"URL Error: {ex.reason}")
        break
    except json.JSONDecodeError as ex:
        print(f"JSON Decoding Error: {ex}")
        break
    except Exception as ex:
        print(f"Error: {ex}")
        break

    # download collective .pgn.gz for each test
    for test, dateStr, meta in ids:
        path = args.path + dateStr + "/" + test + "/"
        if not os.path.exists(args.path + dateStr):
            os.makedirs(args.path + dateStr)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.verbose >= 1:
            print(f"Collecting meta data for test {test} ...")
        if "spsa" in meta.get("args", {}):
            if args.verbose >= 1:
                print(f"  Skipping SPSA test {test} ...")
            continue
        games = None
        if "results" in meta:
            wins = meta["results"].get("wins", 0)
            draws = meta["results"].get("draws", 0)
            losses = meta["results"].get("losses", 0)
            games = wins + draws + losses
            if games == 0:
                if args.verbose >= 1:
                    print(f"  No games found, skipping test {test} ...")
                continue
        tcStrings = None
        if "args" in meta:
            tc = meta["args"].get("tc", "")
            new_tc = meta["args"].get("new_tc", "")
            if tc and new_tc:
                tcStrings = [tc] if tc == new_tc else [tc, new_tc]

        if args.tc_lower_limit is not None or args.tc_upper_limit is not None:
            if tcStrings is None:
                if args.verbose >= 1:
                    print(f"  Missing tc data, skipping test {test} ...")
                continue
            tc_skip = False
            for tc in tcStrings:
                tc_base = re.search(r"^(\d+(\.\d+)?)", tc)
                if tc_base:
                    tc_base = float(tc_base.group(1))
                else:
                    if args.verbose >= 1:
                        print(f'  Malformed tc data "{tc}", skipping test {test} ...')
                    tc_skip = True
                    continue
                if args.tc_lower_limit is not None and tc_base < args.tc_lower_limit:
                    if args.verbose >= 1:
                        print(f'  Too short tc "{tc}", skipping test {test} ...')
                    tc_skip = True
                    continue
                if args.tc_upper_limit is not None and tc_base > args.tc_upper_limit:
                    if args.verbose >= 1:
                        print(f'  Too long tc "{tc}", skipping test {test} ...')
                    tc_skip = True
                    continue
            if tc_skip:
                continue

        url = "https://tests.stockfishchess.org/api/run_pgns/" + test + ".pgn.gz"
        try:
            response = urllib.request.urlopen(url)
            b = response.getheader("Content-Length", None)
            b = "" if b is None else format_large_number(int(b)) + "B "
            msg = f"Downloading {b}.pgn.gz file "
            if games is not None:
                msg += f"with {games} games {'' if args.verbose == 0 else f'(WDL = {wins} {draws} {losses}) '}"
            if args.verbose >= 1 and tcStrings is not None:
                msg += "at TC " + " vs. ".join(tcStrings) + " "
            print(msg + f"to {path} ...")
            tmpName = path + test + ".tmp"
            urllib.request.urlretrieve(url, tmpName)
            os.rename(tmpName, path + test + ".pgn.gz")
            with open(path + test + ".json", "w") as jsonFile:
                json.dump(meta, jsonFile, indent=4, sort_keys=True)
            if args.verbose:
                g = count_games(path + test + ".pgn.gz")
                print(f"Download completed. The file contains {g} games.", end="")
                if games and g < games:
                    print(f" I.e. {games-g} fewer than expected.", end="")
                elif games and g > games:
                    print(f" I.e. {g-games} more than expected.", end="")
                print("")
        except Exception as ex:
            if args.verbose >= 2:
                print(f'  error: caught exception "{ex}"')
            continue
    page += 1  # Move to the next page for the next iteration
print("Finished downloading PGNs.")
