import urllib.request, urllib.error, urllib.parse
import argparse, time, re, os

parser = argparse.ArgumentParser(
    description="Download pgns from completed LTC tests on fishtest.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./",
    help="path to directory in which to store downloaded pgns",
)
parser.add_argument(
    "--subdirs",
    action="store_true",
    help="use PATH/variant/date/test-id/ (sub)directory structure",
)
parser.add_argument(
    "--page",
    type=int,
    default=1,
    help="page of LTC tests to download from",
)
parser.add_argument(
    "--leniency",
    type=int,
    default=3,
    help="one more consecutive HTTP error causes the download of a test to be stopped",
)
args = parser.parse_args()
if args.path == "":
    args.path = "./"
elif args.path[-1] != "/":
    args.path += "/"

paths = [args.path]
if args.subdirs:
    paths += [args.path + "classical", args.path + "frc", args.path + "dfrc"]

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

# find the set of fully downloaded Ids (looking in the full file tree)
p = re.compile("([a-z0-9]*)-0.pgn")  # match only testId-0.pgn
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

# download all pgns...
for test, dateStr in ids:
    if args.subdirs:
        # first check what chess variant was played, based on book name
        variant = ""
        url = "https://tests.stockfishchess.org/tests/view/" + test
        response = urllib.request.urlopen(url)
        webContent = response.read().decode("utf-8").splitlines()
        for i, line in enumerate(webContent):
            if (
                i >= 2
                and webContent[i - 2].endswith("<td>book</td>")
                and (line.endswith(".epd") or line.endswith(".pgn"))
            ):
                for v in ["dfrc", "frc"]:
                    if v in line.lower():
                        variant = v
                        break
                variant = "classical"
                break
        if variant == "":
            variant = "classical"
            print("Could not find book information, assume classical chess.")
        path = args.path + variant + "/" + dateStr + "/" + test + "/"
        if not os.path.exists(args.path + dateStr):
            os.makedirs(args.path + dateStr)
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = args.path
    print(f"""Downloading{"" if args.subdirs else f" {test}'s"} pgns to {path} ...""")
    url = "https://tests.stockfishchess.org/tests/tasks/" + test
    p = re.compile("<a href=/api/pgn/([a-z0-9]*-[0-9]*).pgn>")
    response = urllib.request.urlopen(url)
    webContent = response.read().decode("utf-8").splitlines()
    first, countErrors = True, 0
    for line in reversed(webContent):  # download test-0.pgn last
        m = p.search(line)
        if m:
            filename = m.group(1) + ".pgn"
            if os.path.exists(path + filename):
                countErrors = 0
                continue
        else:
            continue
        time.sleep(0.1)  # be server friendly... wait a bit between requests
        url = "http://tests.stockfishchess.org/api/pgn/" + filename
        if first:
            _, _, number = m.group(1).partition("-")
            print(f"  Fetching {int(number)+1} missing pgns ...")
            first = False
        try:
            tmpName = test + ".tmp"
            urllib.request.urlretrieve(url, path + tmpName)
            os.rename(path + tmpName, path + filename)
            countErrors = 0
        except urllib.error.HTTPError as error:
            print(f"  HTTP Error {error.code} occurred for URL: {url}")
            countErrors += 1
            if countErrors > args.leniency:
                print(f"  Skipping remaining pgns of test {test} ...")
                break
        except Exception as ex:
            print(f'  error: caught exception "{ex}"')
            continue
