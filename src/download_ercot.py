from pathlib import Path
import requests
import zipfile
import io
import json

RAW = Path("data/raw/ercot_dam_prices")
RAW.mkdir(parents=True, exist_ok=True)

REPORT_TYPE_ID = "13060"

LIST_URL = (
    "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
    f"?reportTypeId={REPORT_TYPE_ID}"
)

DOWNLOAD_URL = (
    "https://www.ercot.com/misdownload/servlets/mirDownload"
    "?doclookupId={doc_id}"
)

def get_report_list():
    r = requests.get(LIST_URL, timeout=60)
    r.raise_for_status()

    data = r.json()

    with open(RAW / "ercot_report_list.json", "w") as f:
        json.dump(data, f, indent=2)

    return data

def extract_doc_ids(data):
    """
    ERCOT JSON structure can vary slightly, so this searches recursively
    for keys that look like doc IDs.
    """
    doc_ids = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in {"docid", "doc_id", "doclookupid", "doclookup_id"}:
                    doc_ids.append(str(v))
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    return sorted(set(doc_ids))

def download_doc(doc_id):
    url = DOWNLOAD_URL.format(doc_id=doc_id)
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    content = r.content

    try:
        z = zipfile.ZipFile(io.BytesIO(content))
        z.extractall(RAW / doc_id)
        print(f"Extracted ZIP for doclookupId={doc_id}")
    except zipfile.BadZipFile:
        out = RAW / f"{doc_id}.xlsx"
        out.write_bytes(content)
        print(f"Saved file for doclookupId={doc_id}: {out}")

def main():
    data = get_report_list()
    doc_ids = extract_doc_ids(data)

    print(f"Found {len(doc_ids)} document IDs")

    if not doc_ids:
        print("No doc IDs found. Open data/raw/ercot_dam_prices/ercot_report_list.json and inspect the structure.")
        return

    for doc_id in doc_ids:
        download_doc(doc_id)

if __name__ == "__main__":
    main()