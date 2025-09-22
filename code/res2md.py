import sys
import json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python res2md.py <result_file>")
        sys.exit(1)

    result_file = sys.argv[1]
    with open(result_file, "r") as f:
        data = f.readlines()
        if len(data) == 0:
            print("No data found in the file")
            sys.exit(1)
        data = data[-1].strip()

        j = json.loads(data)
        print(j)
