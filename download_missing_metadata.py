import argparse, json, os, re, requests, tarfile, time, urllib.request

parser = argparse.ArgumentParser(
    description="Download fishtest metadata for any test that .pgn(.gz) files can be found for.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--path",
    default="./pgns",
    help="Recursively look for .pgn(.gz) files in this directory.",
)
parser.add_argument(
    "-o",
    "--overwrite",
    action="store_true",
    help="Overwrite existing metadata (use with care!).",
)
args = parser.parse_args()

if not os.path.exists(args.path):
    print(f"Error: directory {args.path} not found.")
    exit

# find the set of downloaded Ids (looking in the full file tree)
p = re.compile("([a-z0-9]*)-[0-9]*.pgn(|.gz)")
tests = set()

for path, _, files in os.walk(args.path):
    for name in files:
        m = p.match(name)
        if m:
            full_path = os.path.join(path, m.group(1) + ".json")
            tests.add(full_path)

print(f"Found {len(tests)} downloaded tests in {args.path}.")

p = re.compile("([a-z0-9]*).json")

# download metadata for each test
for json_name in tests:
    if os.path.exists(json_name) and not args.overwrite:
        print(f"File {json_name} exists already, skipping download.")
        continue

    test = p.match(os.path.basename(json_name)).group(1)
    url = "https://tests.stockfishchess.org/api/get_run/" + test
    try:
        meta = requests.get(url).json()
        with open(json_name, "w") as jsonFile:
            json.dump(meta, jsonFile, indent=4, sort_keys=True)
        print(f"Downloaded {json_name}.")
        time.sleep(0.1)  # be server friendly... wait a bit between requests
    except Exception as ex:
        print(f'  error: caught exception "{ex}"')
