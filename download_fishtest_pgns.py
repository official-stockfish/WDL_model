import urllib.request, urllib.error, urllib.parse
import argparse, time, re, os, json

parser = argparse.ArgumentParser(
    description="Download pgns from completed LTC tests on fishtest.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./pgns",
    help="Downloaded pgns will be stored in PATH/date/test-id/.",
)
parser.add_argument(
    "--page",
    type=int,
    default=1,
    help="Page of completed LTC tests to download from.",
)
parser.add_argument(
    "--leniency",
    type=int,
    default=3,
    help="One more consecutive HTTP error causes the download of a test to be stopped.",
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
p = re.compile("([a-z0-9]*)-0.pgn(|.gz)")  # match only testId-0.pgn(.gz)
downloaded = set()

for _, _, files in os.walk(args.path):
    for name in files:
        m = p.match(name)
        if m:
            downloaded.add(m.group(1))

print(f"Found {len(downloaded)} fully downloaded tests in {args.path} already.")

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

# download all pgns, together with some of the tests' meta data...
for test, dateStr in ids:
    url = "https://tests.stockfishchess.org/tests/view/" + test
    response = urllib.request.urlopen(url)
    webContent = response.read().decode("utf-8").splitlines()
    if "<td>spsa</td>" in "".join(webContent):
        if args.verbose >= 1:
            print(f"Skipping SPSA test {test} ...")
        continue
    path = args.path + dateStr + "/" + test + "/"
    if not os.path.exists(args.path + dateStr):
        os.makedirs(args.path + dateStr)
    if not os.path.exists(path):
        os.makedirs(path)
    if args.verbose >= 1:
        print(f"Collecting meta data for test {test} ...")
    meta = {}
    keyStrs = [
        "adjudication",  # first the keywords that have the value on next but one line
        "base_net",
        "base_options",
        "base_tag",
        "book",
        "book_depth",
        "new_net",
        "new_options",
        "new_tag",
        "new_tc",
        "sprt",
        "tc",
        "threads",
        "start time",  # then the keywords that appear on the same line as the value
        "last updated",
    ]
    p = re.compile("<td>([0-9 :\-]*)</td>")
    for i, line in enumerate(webContent):
        if i < 2:
            continue
        for keyStr in keyStrs[:-2]:
            if webContent[i - 2].endswith(f"<td>{keyStr}</td>"):
                meta[keyStr] = line.strip()
        for keyStr in keyStrs[-2:]:
            if keyStr in line:
                meta[keyStr] = p.search(line).group(1)
    for keyStr in keyStrs:
        if keyStr not in meta:
            if args.verbose >= 2:
                print(f"Could not find {keyStr} information at {url}.")
    with open(path + test + ".json", "w") as jsonFile:
        json.dump(meta, jsonFile, indent=4, sort_keys=True)

    print(f"Downloading pgns to {path} ...")
    url = "https://tests.stockfishchess.org/tests/tasks/" + test
    p = re.compile("<a href=/api/pgn/([a-z0-9]*-[0-9]*).pgn>")
    response = urllib.request.urlopen(url)
    webContent = response.read().decode("utf-8").splitlines()
    first, countErrors = True, 0
    for line in reversed(webContent):  # download test-0.pgn last
        m = p.search(line)
        if m:
            filename = m.group(1) + ".pgn"
            if os.path.exists(path + filename) or os.path.exists(path + filename + ".gz"):
                countErrors = 0
                continue
        else:
            continue
        time.sleep(0.1)  # be server friendly... wait a bit between requests
        url = "http://tests.stockfishchess.org/api/pgn/" + filename
        if first:
            _, _, number = m.group(1).partition("-")
            if args.verbose >= 1:
                print(f"  Fetching {int(number)+1} missing pgns ...")
            first = False
        try:
            tmpName = test + ".tmp"
            urllib.request.urlretrieve(url, path + tmpName)
            os.rename(path + tmpName, path + filename)
            countErrors = 0
        except urllib.error.HTTPError as error:
            if args.verbose >= 2:
                print(f"  HTTP Error {error.code} occurred for URL: {url}")
            countErrors += 1
            if countErrors > args.leniency:
                if args.verbose >= 2:
                    print(f"  Skipping remaining pgns of test {test} ...")
                break
        except Exception as ex:
            if args.verbose >= 2:
                print(f'  error: caught exception "{ex}"')
            continue
