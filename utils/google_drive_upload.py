# utils/google_drive_upload.py

import glob
import os

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def upload_to_gdrive(local_path, gdrive_folder_id=None):
    # Søg efter credentials/secret
    secret_path = os.path.join(
        os.path.dirname(__file__),
        "../config/client_secret_446109686324-b2knvjhlr7q91g5ds22latp9gj88c85i.apps.googleusercontent.com.json",
    )  # Ret filnavn hvis nødvendigt
    secret_path = os.path.abspath(secret_path)

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(secret_path)
    gauth.LocalWebserverAuth()  # Første gang: åbner browser til login
    drive = GoogleDrive(gauth)

    file_name = os.path.basename(local_path)
    file_metadata = {"title": file_name}
    if gdrive_folder_id:
        file_metadata["parents"] = [{"id": gdrive_folder_id}]
    file = drive.CreateFile(file_metadata)
    file.SetContentFile(local_path)
    file.Upload()
    print(
        f"✅ Uploadet til Google Drive: {file_name} (folder: {gdrive_folder_id if gdrive_folder_id else 'root'})"
    )


if __name__ == "__main__":
    # --- Find de nyeste outputfiler ---
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs/walkforward/")
    output_dir = os.path.abspath(output_dir)
    extensions = ["csv", "xlsx", "json"]
    uploaded = 0

    for ext in extensions:
        files = sorted(
            glob.glob(os.path.join(output_dir, f"walkforward_summary*.{ext}")),
            key=os.path.getmtime,
            reverse=True,
        )
        for f in files[:3]:  # Upload de 3 nyeste af hver type
            upload_to_gdrive(f)
            uploaded += 1

    # --- Find og upload evt. top-5/top-10 splits ---
    for pattern in ["top5_*.csv", "top10_*.csv"]:
        files = sorted(
            glob.glob(os.path.join(output_dir, pattern)),
            key=os.path.getmtime,
            reverse=True,
        )
        for f in files[:2]:
            upload_to_gdrive(f)
            uploaded += 1

    if uploaded == 0:
        print("Ingen relevante walkforward-resultater fundet til upload.")
