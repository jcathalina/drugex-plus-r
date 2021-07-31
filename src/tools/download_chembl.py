import requests
import argparse
import os
import sys

from tqdm import tqdm


FILES = {
    "chembl_26_sdf": {
        "filename": "chembl_26.sdf.gz",
        "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_26/chembl_26.sdf.gz"
    }
}


def download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True, verify=False) as response:
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))
        pbar = tqdm(total=file_size, desc=os.path.basename(filename), unit='B', unit_scale=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser("download_raw_data")
    parser.add_argument(
        "path",
        default="data/raw",
        help="the path that you want to download the chembl data file to."
    )
    path = parser.parse_args().path

    try:
        for file in FILES.values():
            download_file(url=file["url"], filename=file["filename"])
    except requests.HTTPError as http_error:
        print(f"Download has failed: {str(http_error)}")
        sys.exit(1)
    
    print("Files have been successfully downloaded.")
