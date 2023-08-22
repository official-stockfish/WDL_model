import urllib.request, urllib.error, urllib.parse
import argparse, time, re, os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    default="./",
    help="path to directory in which to store downloaded pgns",
)
args = parser.parse_args()
if args.path == "":
    args.path = "./"
elif args.path[-1] != "/":
    args.path += "/"

print(f"Download pgns to directory {args.path} ...")

# find the set of already downloaded Ids (looking in the full file tree)
p = re.compile("([a-z0-9]*)-[0-9]*.pgn")
downloaded = set()

for path, subdirs, files in os.walk(args.path):
    for name in files:
        m = p.match(name)
        if m:
            downloaded.add(m.group(1))

print(f"Found {len(downloaded)} downloaded tests already.")

# fetch page with finished LTC tests, parse and list new IDs
url = "https://tests.stockfishchess.org/tests/finished?ltc_only=1"
p = re.compile('<a href="/tests/view/([a-z0-9]*)">')

ids = []
response = urllib.request.urlopen(url)
webContent = response.read().decode("utf-8").splitlines()
for line in webContent:
    m = p.search(line)
    if m:
        testId = m.group(1)
        if not testId in downloaded:
            ids.append(testId)

# download all pgns...
for test in ids:
    print(f"Downloading test {test} ...")
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
            response = urllib.request.urlopen(url)
        except:
            continue
        else:
            webContent = response.read()
            try:
                f = open(args.path + filename, "w")
                f.write(webContent.decode("utf-8"))
            except:  # if itf-8 encoding fails, fall back to latin-1
                f.close
                f = open(args.path + filename, "w")
                f.write(args.path + webContent.decode("latin-1"))
            f.close
