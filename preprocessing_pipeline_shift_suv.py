# Allows us to download from Hugging Face 
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi
# Hopefully convert to Zarr with Libraries
import noether 
import noether.data.zarr_store
# Read in stl, vtp and convert to tensors
from physicsnemo.datapipes.readers.vtk import VTKReader
import pyvista as pv 
import os 
#  
import shutil 
import numpy as np
from pathlib import Path

# Get Dataset from Hugging Face (Shift-SUV Sample)
REPO_ID = "luminary-shift/SUV-sample"

#PRESSURE = 

# Files we want to preprocess into ZARRs / PT Tensors
FILES_TO_DOWNLOAD = [
    # Merged_surfaces.stl - STL file with Vehicles Geometry 
    "merged_surfaces.stl",
    # merged_surfaces.vtp - Surface field solution file
    "merged_surfaces.vtp",
    # Could use to calculate Drag
    "DRAG_COEFFICIENT.csv",
    # Could use to calculate Lift
    "LIFT_COEFFICIENT.csv"
]

'''
vtp files for Shift-SUV are:
['vtkOriginalPointIds', 'Normals', 'pressure', 'pressure_average',
'wall_shear_stress', 'wall_shear_stress_average',
'vtkOriginalCellIds', 'Normals']
'''
SURFACE_VTP_KEYS = [
    "pressure_average",
    "wall_shear_stress_average",
    "Normals"
]

'''
stl files for Shift-SUV are:
['stl_faces', 'stl_coordinates', 'surface_normals', 'stl_centers', 'stl_areas']
'''
SURFACE_STL_KEYS = [

]

# Gets the Names of the dataset files without download: 
# 2220 Files, Cases split into folders with prefix by RUNS_X from 1 - 100)
files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
runs = sorted({f.split("/")[0] for f in files if f.startswith("RUN_")},
            key=lambda r: int(r.split("_")[1]))

runs = runs[:1]

import pyvista as pv, numpy as np
from scipy.spatial import cKDTree

mesh = pv.read(r"C:\Users\Svn15\Dissertation_Project\temp_raw_shift_suv\RUN_1\merged_surfaces.vtp")
reader = VTKReader(
                    path=fr"{Path.cwd()}/temp_raw_shift_suv",
                    #keys_to_read=['pressure_average', 'wall_shear_stress_average', 'Normals']
                    )
#print(reader.field_names)

data, metadata = reader[0]

stl_xyz = data["stl_coordinates"].numpy()

# Read in File for vtp file

data = pv.read(filename=r"C:\Users\Svn15\Dissertation_Project\temp_raw_shift_suv\RUN_1\merged_surfaces.vtp")
print(f"Shape of Normal is: {data["Normals"].shape}")
print(f"Shape of Averaged Pressure is: {data["pressure_average"].shape}")
print(f"Shape of Averaged Wall_Shear_Stress_average is: {data["wall_shear_stress_average"].shape}")
print(f"Shape of Surface Positions is: {(data.cell_centers().points).shape}")

#print(data.n_points, data.n_cells)
#print(data.is_all_triangles)
#print(list(data.cell_data.keys()))  


#print(data.array_names)

# Read in File for stl file 

#reader = VTKReader(
#                    path=f"{Path.cwd()}/temp_raw_shift_suv",
#                    #keys_to_read=['pressure_average', 'wall_shear_stress_average', 'Normals']
#                    )
#print(reader.field_names)

#data, metadata = reader[0]
#print(f"Shape of stl_coordinates is: {data["stl_coordinates"].shape}")
#print(f"Shape of stl_faces is: {data["stl_faces"].shape}")
#print(f"Shape of stl_normals is: {data["surface_normals"].shape}")
#print(f"Shape of stl_centers is: {data["stl_centers"].shape}")
#print(f"Shape of stl_areas is: {data["stl_areas"].shape}")





#for run in runs:
 #   for filename in FILES_TO_DOWNLOAD:
  #      print(f"Starting download: {run}/{filename}")
       # try:
            # Downloads to cache and returns the local file path
        #    local_path = hf_hub_download(
         #       repo_id=REPO_ID,
          #      repo_type="dataset",
           #     filename=fr"{run}/{filename}",
            #    local_dir=fr"{Path.cwd()}/temp_raw_shift_suv"  # Optional: force download to a specific folder
            #)
            #print(f"Successfully saved to: {local_path}\n")

       # except HfHubHTTPError as e:
       #     print(f"Failed to download {filename}: {e}\n")
            # You can add retry logic here if needed
        
       # print(f"Starting Preprocessing to Tensor: {run}/{filename}")
        #Preprocessing Step 

        # Check if csv file or stl/vtp
        # Read in Raw Data to a reader 
    
       # reader = VTKReader(
       #     path=f"{Path.cwd()}/temp_raw_shift_suv",
            #keys_to_read=["stl_coordinates", "stl_faces", "point_data", "cell_data"]
       # )
        

        # Convert to Tensors 
        #print(reader._get_field_names())

        # Delete File After Processed
        # os.remove(f"{local_path}")

