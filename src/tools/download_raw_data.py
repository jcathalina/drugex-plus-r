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


def download_file(url: str, filename: str, destdir: str) -> None:
    with requests.get(url, stream=True, verify=False) as response:
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))
        pbar = tqdm(total=file_size, desc=os.path.basename(filename), unit='B', unit_scale=True)
        with open(os.path.join(destdir, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    # TODO: Fix argparse bugs. 1) Does not default to data/raw, is not optional... 2) Download gets stuck for some reason
    parser = argparse.ArgumentParser(prog="download_raw_data",
                                     description="Downloads all the necessary raw data files for DrugEx to work.")
    parser.add_argument(
        "destdir",
        default="data/raw",
    )

    destdir = parser.parse_args().destdir

    try:
        for file in FILES.values():
            download_file(url=file["url"], filename=file["filename"], destdir=destdir)
    except requests.HTTPError as http_error:
        print(f"Download has failed: {str(http_error)}")
        sys.exit(1)
    
    print("Files have been successfully downloaded.")


if __name__ == "__main__":
    main()
