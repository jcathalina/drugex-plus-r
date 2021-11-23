import os

import click
import requests
from pyprojroot import here
from tqdm import tqdm

VALID_VERSIONS = [26, 27, 28, 29]


def _get_chembl_url(version: int) -> str:
    url = f"http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{version}/chembl_{version}.sdf.gz"
    return url


def _get_chembl_filename(version: int) -> str:
    filename = f"chembl_{version}.sdf.gz"
    return filename


def download_file(url: str, filename: str, destdir: os.PathLike) -> None:
    with requests.get(url, stream=True, verify=False) as response:
        response.raise_for_status()
        file_size = int(response.headers.get("content-length", 0))
        pbar = tqdm(
            total=file_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(os.path.join(destdir, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


@click.command()
@click.option(
    "--destdir",
    default=f"{here('data/raw')}",
    help="destination directory for downloaded ChEMBL files",
)
@click.option("--version", default=26, help="version of ChEMBL to download")
def download_chembl_data(destdir: os.PathLike, version: int) -> None:
    """
    Downloads the compressed ChEMBL database file given a valid version number
    :param destdir: Directory to download ChEMBL file to
    :param version: The version number for the ChEMBL database file
    """
    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Version {version} of ChEMBL is not supported or does not exist."
        )

    click.echo(f"Downloading ChEMBL v{version} data")
    url = _get_chembl_url(version=version)
    filename = _get_chembl_filename(version=version)

    try:
        download_file(url=url, filename=filename, destdir=destdir)
    except requests.HTTPError as http_error:
        print(f"Download has failed: {str(http_error)}")
