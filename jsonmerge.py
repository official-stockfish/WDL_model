import argparse, json


def merge_file(filename, merged_data):
    with open(filename) as file:
        data = json.load(file)
        for key, value in data.items():
            merged_data[key] = merged_data.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser(description="Merge several JSON files.")
    parser.add_argument("input", nargs="+", help="JSON files to merge")
    parser.add_argument(
        "-o", "--output", default="merged_data.json", help="merged JSON file"
    )
    args = parser.parse_args()

    merged_data = {}
    for filename in args.input:
        merge_file(filename, merged_data)

    with open(args.output, "w") as output_file:
        json.dump(merged_data, output_file, indent=4)

    print(f"Merging JSON file saved to {args.output}.")


if __name__ == "__main__":
    main()
