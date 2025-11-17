import time
import kagglehub
import sys

class KagglehubDownloader:
    """
    A class to handle dataset downloading from Kaggle using kagglehub with progress status.
    """
    @staticmethod
    def download_with_status(dataset_name: str = "fedesoriano/air-quality-data-set") -> str:
        """
        Wrapper around kagglehub.dataset_download with simple text progress.
        This does NOT show real bytes downloaded, but gives the user feedback.
        """
        print(f"Preparing to download dataset: {dataset_name}")
        path = kagglehub.dataset_download(dataset_name)

        print("Download complete.")
        print("Path to dataset files:", path)
        return path