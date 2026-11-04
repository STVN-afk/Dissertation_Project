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
import torch
from pathlib import Path

# Get Dataset from Hugging Face (Shift-SUV Sample)
REPO_ID = "luminary-shift/SUV-sample"
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

# Pascal-Seconds
DYNAMIC_VISCOCITY = 1.81e-5 
# Temperature in Kelvin (K)
STATIC_TEMP = 293.15
# Fluid Density in kg/m^3
DENSITY = 1.225
# Yaw in Degrees
YAW = 0.0
# Angle Of Attack in Degrees
ANGLE_OF_ATTACK = 0.0
# Inflow Velocity (Known as Freestream Velocity) in m/s 
IF_VELO = 50

# Freestream pressure used for creating coefficients to make data independent from flow speed, fluid density and scale. 
FREESTREAM_DYNAMIC_PRESSURE = 1/2 * DENSITY * (IF_VELO ** 2)

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
     "stl_areas",
     "surface_normals",
     "stl_centers"
]

# Gets the Names of the dataset files without download: 
# 2220 Files, Cases split into folders with prefix by RUNS_X from 1 - 100)
files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
runs = sorted({f.split("/")[0] for f in files if f.startswith("RUN_")},
            key=lambda r: int(r.split("_")[1]))

runs = runs[:1]

import pyvista as pv, numpy as np
from scipy.spatial import cKDTree

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



output_tensor_path = Path("Data/processed_shiftsuv")

if output_tensor_path.exists():
     print("Path for processed SHIFT-SUV data is already created, starting preprocessing")
else:
     output_tensor_path.parent.mkdir(parents=True, exist_ok=True)


for run in runs:
     for filename in FILES_TO_DOWNLOAD:
          print(f"Starting download: {run}/{filename}")
          try:
               # Downloads to cache and returns the local file path
               local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=fr"{run}/{filename}",
                    local_dir=fr"{Path.cwd()}/temp_raw_shift_suv"  # Optional: force download to a specific folder
               )
               print(f"Successfully saved to: {local_path}\n")

          except HfHubHTTPError as e:
               print(f"Failed to download {filename}: {e}\n")
               # You can add retry logic here if needed

          #Preprocessing Step 
          # Check if csv file or stl/vtp
          if '.vtp' in filename:
               print(f"Starting Preprocessing vtp file to Tensor: {run}/{filename}")
               data = pv.read(filename=fr"{Path.cwd()}/temp_raw_shift_suv/{run}/{filename}")

               # p_inf is the freestream pressure computed from total pressure and dynamic pressure 
               # Accurate baseline for sea-level static pressure
               p_inf = data["pressure_average"].max() - FREESTREAM_DYNAMIC_PRESSURE

               # Pressure coefficient to make aerodynamic data independent from flow speed, fluid density and scale.
               PRESSURE_COEFFICIENT = (data["pressure_average"] - p_inf) / FREESTREAM_DYNAMIC_PRESSURE

               # Wall Shear coefficient making data independent from flow speed, fluid density and scale.
               WALL_SHEAR_STRESS_COEFFICIENT = data["wall_shear_stress_average"] / FREESTREAM_DYNAMIC_PRESSURE

               # Non-dimensionizing Surface position vehicle Mesh through normalization because size of SHIFT-SUV is 1/4 scale compared to DrivAerML.
               position_data = data.cell_centers().points
               x_min, y_min, z_min = position_data.min(axis=0)
               x_max, y_max, z_max = position_data.max(axis=0)

               # Vehicle Length for normalization 
               VEHICLE_LEN = x_max - x_min

               x_origin = x_min
               y_origin = (y_min + y_max) / 2.0
               z_origin = z_min

               # Normalize data to make it scale-invariant, 
               pos_normalized = np.zeros_like(position_data)
               pos_normalized[:, 0] = (position_data[:, 0] - x_min) / VEHICLE_LEN     # X* in [0.0, 1.0]
               pos_normalized[:, 1] = (position_data[:, 1] - y_origin) / VEHICLE_LEN  # Y* centered at 0.0
               pos_normalized[:, 2] = (position_data[:, 2] - z_min) / VEHICLE_LEN     # Z* ground plane at 0.0

               # Saving to tensors at specific location
               output_case_path = Path(f"{output_tensor_path}/{run}")

               # If it does not exist, create a new folder for RUN
               if not output_case_path.exists():
                    output_case_path.mkdir(parents=True, exist_ok=True)

               # Convert pressure tensor from [n] to [n,1]
               pressure_tensor = torch.tensor(PRESSURE_COEFFICIENT, dtype=torch.float32).unsqueeze(-1)
               torch.save(pressure_tensor,f"{output_case_path}/pressure_tensors.pt")
               wallshear_tensor = torch.tensor(WALL_SHEAR_STRESS_COEFFICIENT, dtype=torch.float32)
               torch.save(wallshear_tensor,f"{output_case_path}/wallshear_tensors.pt")
               surface_pos_tensor = torch.tensor(pos_normalized, dtype=torch.float32)
               torch.save(surface_pos_tensor,f"{output_case_path}/surface_pos_tensors.pt")

          # Read in Raw stl Data to a reader 
          elif '.stl' in filename:
               reader = VTKReader(
               path=f"{Path.cwd()}/temp_raw_shift_suv/",
               keys_to_read=SURFACE_STL_KEYS
               )

               print(reader.field_names)
               data, metadata = reader[0]

               print(f"Shape of stl_normals is: {data["surface_normals"].shape}")
               print(f"Shape of stl_areas is: {data["stl_areas"].shape}")
               print(f"Shape of stl_centers is: {data["stl_centers"].shape}")

               # Non-dimensionizing Geometry vehicle Mesh through normalization because size of SHIFT-SUV is 1/4 scale compared to DrivAerML.
               geom_position_data = data["stl_centers"]
               mins = geom_position_data.min(axis=0).values
               maxs = geom_position_data.max(axis=0).values
               x_min, y_min, z_min = mins[0].item(),mins[1].item(),mins[2].item()
               x_max, y_max, z_max = maxs[0].item(),maxs[1].item(),maxs[2].item()


               # Vehicle Length for normalization 
               VEHICLE_LEN = x_max - x_min

               x_origin = x_min
               y_origin = (y_min + y_max) / 2.0
               z_origin = z_min

               # Normalize data to make it scale-invariant, 
               geometry_normalized = np.zeros_like(geom_position_data)
               geometry_normalized[:, 0] = (geom_position_data[:, 0] - x_min) / VEHICLE_LEN     # X* in [0.0, 1.0]
               geometry_normalized[:, 1] = (geom_position_data[:, 1] - y_origin) / VEHICLE_LEN  # Y* centered at 0.0
               geometry_normalized[:, 2] = (geom_position_data[:, 2] - z_min) / VEHICLE_LEN     # Z* ground plane at 0.0

               # Normalizes the areas to match the normalized version of the vehicle surface mesh.
               geom_areas_normalized = data["stl_areas"] / (VEHICLE_LEN **2)


               output_case_path = Path(f"{output_tensor_path}/{run}")
               # If it does not exist, create a new folder for RUN
               if not output_case_path.exists():
                    output_case_path.mkdir(parents=True, exist_ok=True)

               geom_pos_tensor = torch.tensor(geometry_normalized, dtype=torch.float32)
               torch.save(geom_pos_tensor,f"{output_case_path}/geom_pos_tensors.pt")
               geom_area_tensor = torch.tensor(geom_areas_normalized, dtype=torch.float32)
               torch.save(geom_area_tensor,f"{output_case_path}/geom_area_tensors.pt")

          # Delete File After Processed
          os.remove(f"{local_path}")

     # Deletes File 
     #shutil.rmtree(f"temp_raw_shift_suv/{run}")

