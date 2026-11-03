# Allows us to download from Hugging Face 
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi
# Hopefully convert to Zarr with Libraries
import noether 
import noether.data.zarr_store
# Physics-Nemo to read in the raw stl/vtp files 
from physicsnemo.datapipes.readers.vtk import VTKReader

#  
import shutil 
from pathlib import Path

# Get Dataset from Hugging Face (DrivAerML)
REPO_ID = "neashton/drivaerml"

# Files we want to preprocess into ZARRs / PT Tensors
FILES_TO_DOWNLOAD = [
    # drivaer_i.stl - STL file with runs Geometry 
    "drivaer_x.stl",
    # boundary_i.vtp - Surface field solution file
    "boundary_x.vtp",
    # boundary_cell_are_i.npy - the area of each surface cell (Could be useful for area-weighted losses / integrating pressure into drag)
    "boundary_cell_area_x.npy",
    # forces/moments time-averaged
    "force_mom_constref_x.csv",
    # reference values for each geometry
    "geo_ref_x.csv",
    # reference geometry for each geometry
    "geo_parameters_x.csv"
]

# Gets the Names of the dataset files without download: 
# 2220 Files, Cases split into folders with prefix by RUNS_X from 1 - 442)
files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
runs = sorted({f.split("/")[0] for f in files if f.startswith("run")},
            key=lambda r: int(r.split("_")[1]))

# Adjust this to change number of datasets to process
runs = runs[:1]

# Enumerate since files include the run number 
for idx,run in enumerate(runs,1):
    for filename in FILES_TO_DOWNLOAD:
        # Replaces x with run number 
        filename = filename.replace("x",f"{idx}")
        print(f"Starting download: {run}/{filename}")
        try:
            # Downloads to cache and returns the local file path
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{run}/{filename}",
                local_dir=fr"{Path.cwd()}/temp_raw_drivAerml"  
            )
            print(f"Successfully saved to: {local_path}\n")

            # Preprocessing Steps Here 
            reader = VTKReader(
            path=f"{Path.cwd()}/temp_raw_drivAerml/{run}/{filename}",
            #keys_to_read=["stl_coordinates", "stl_faces", "point_data", "cell_data"]
            )

            print(reader._get_field_names)
            break


            # Delete RAW FILE AFTER
    
        except HfHubHTTPError as e:
            print(f"Failed to download {filename}: {e}\n")
            # You can add retry logic here if needed
