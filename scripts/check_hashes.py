from hashlib import sha256
from pathlib import Path

raw_folder = Path(__file__).resolve().parents[1] / "data" / "raw"

for filename in ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]:
    file_path = raw_folder / filename
    fingerprint = sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            fingerprint.update(chunk)

    print(filename, fingerprint.hexdigest())