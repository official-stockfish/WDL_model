import urllib.request, urllib.error, urllib.parse
import argparse, time, re, os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    default="./",
    help="path to directory in which to store downloaded pgns",
)
parser.add_argument(
    "--subdirs",
    action="store_true",
    help="use PATH/date/test-id/ (sub)directory structure",
)
parser.add_argument(
    "--page",
    type=int,
    default=1,
    help="page of LTC tests to download from",
)
args = parser.parse_args()
if args.path == "":
    args.path = "./"
elif args.path[-1] != "/":
    args.path += "/"

if not os.path.exists(args.path):
    os.makedirs(args.path)

# find the set of already downloaded Ids (looking in the full file tree)
# a test is considered downloaded if at least one of its pgn was (partially) downloaded
# TODO: we may want to download missing pgn's for partially downloaded tests in future
p = re.compile("([a-z0-9]*)-[0-9]*.pgn")
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

# download all pgns...
for test, dateStr in ids:
    if args.subdirs:
        path = args.path + dateStr + "/" + test + "/"
        if not os.path.exists(args.path + dateStr):
            os.makedirs(args.path + dateStr)
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = args.path
    print(f"""Downloading{"" if args.subdirs else f" {test}'s"} pgns to {path} ...""")
    url = "https://tests.stockfishchess.org/tests/tasks/" + test
    p = re.compile("<a href=/api/pgn/([a-z0-9]*-[0-9]*.pgn)>")
    response = urllib.request.urlopen(url)
    webContent = response.read().decode("utf-8").splitlines()
    for line in webContent:
        m = p.search(line)
        if m:
            filename = m.group(1)
        else:
            continue
        time.sleep(0.1)  # be server friendly... wait a bit between requests
        url = "http://tests.stockfishchess.org/api/pgn/" + filename
        try:
            urllib.request.urlretrieve(url, path + filename)
        except:
            continue
