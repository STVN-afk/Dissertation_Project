# Allows us to download from Hugging Face 
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi
# Hopefully convert to Zarr with Libraries
import noether 
import noether.data.zarr_store
# Physics-Nemo to read in the raw stl/vtp files 
# Read in stl, vtp and convert to tensors
from physicsnemo.datapipes.readers.vtk import VTKReader
import pyvista as pv 
import os 
#  
import shutil 
import numpy as np
import torch
from pathlib import Path
import pandas as pd 

# Pascal-Seconds - Calulated from kinematic viscocity x air density 
DYNAMIC_VISCOCITY = 1.8146e-5
# Temperature in Kelvin (K)
STATIC_TEMP = 293.15
# Fluid Density in kg/m^3 through the ideal gas law as stated by paper from normalization of 
# force coefficients
DENSITY = 1.0
# Yaw in Degrees
YAW = 0.0
# Angle Of Attack in Degrees
ANGLE_OF_ATTACK = 0.0
# Inflow Velocity (Known as Freestream Velocity) in m/s 
IF_VELO = 38.889

# Freestream pressure used for creating coefficients to make data independent from flow speed, fluid density and scale.
FREESTREAM_KINEMATIC_PRESSURE = 1/2 * (IF_VELO ** 2)

# Get Dataset from Hugging Face (DrivAerML)
REPO_ID = "neashton/drivaerml"

# Files we want to preprocess into ZARRs / PT Tensors
FILES_TO_DOWNLOAD = [
    # drivaer_i.stl - STL file with runs Geometry 
    "drivaer_x.stl",
    # boundary_i.vtp - Surface field solution file
    # Contains
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

'''
vtp files for DrivAerml are:
['TimeValue', 'CpMeanTrim', 'pMeanTrim', 'pPrime2MeanTrim', 'wallShearStressMeanTrim']
'''
SURFACE_VTP_KEYS = [
    "pMeanTrim",
    "wallShearStressMeanTrim",
]

'''
stl files for DrivAerml are:
'''
SURFACE_STL_KEYS = ["stl_coordinates", 
                    "stl_areas",
                    "stl_centers"]

# Gets the Names of the dataset files without download: 
# 2220 Files, Cases split into folders with prefix by RUNS_X from 1 - 442)
files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
runs = sorted({f.split("/")[0] for f in files if f.startswith("run")},
            key=lambda r: int(r.split("_")[1]))

# Adjust this to change number of datasets to process
runs = runs[:10]

output_tensor_path = Path("Data/processed_drivAerML")

'''# --- How to convert to stl/vtp file ---
point_cloud = pv.PolyData(geometry_normalized)
point_cloud["stl_areas"] = geom_areas_normalized
# 4. Export using binary VTP format
output_vtp = Path("./drivaerml_normalized_geometry_from_stl_only.vtp")
point_cloud.save(str(output_vtp), binary=True)'''







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

            if '.vtp' in filename:
                data = pv.read(filename=fr"{Path.cwd()}/temp_raw_drivAerml/{run}/{filename}")

                # p_inf is the freestream pressure computed from total pressure and dynamic pressure 
                # Accurate baseline for sea-level static pressure
                p_inf = data["pMeanTrim"].max() - FREESTREAM_KINEMATIC_PRESSURE
            
                # Use Pressure Coefficent Given by Dataset 
                # Pressure coefficient to make aerodynamic data independent from flow speed, fluid density and scale.
                PRESSURE_COEFFICIENT = data["CpMeanTrim"]

                # Wall Shear coefficient making data independent from flow speed, fluid density and scale.
                WALL_SHEAR_STRESS_COEFFICIENT = data["wallShearStressMeanTrim"] / FREESTREAM_KINEMATIC_PRESSURE

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

                print(pos_normalized.shape)
                # Saving to tensors at specific location
                output_case_path = Path(f"{output_tensor_path}/{run.upper()}")

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

            elif '.stl' in filename:
                reader = VTKReader(
                    path=f"{Path.cwd()}/temp_raw_drivAerml/",
                    keys_to_read=SURFACE_STL_KEYS
                )

                data, metadata = reader[0]
                stl_areas = data['stl_areas']
                print(f"Stl Area Shape: {stl_areas.shape}")
                stl_centers = data['stl_centers']
                print(f"Stl Centers Shape: {stl_centers.shape}")

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


                output_case_path = Path(f"{output_tensor_path}/{run.upper()}")
                # If it does not exist, create a new folder for RUN
                if not output_case_path.exists():
                    output_case_path.mkdir(parents=True, exist_ok=True)

                geom_pos_tensor = torch.tensor(geometry_normalized, dtype=torch.float32)
                torch.save(geom_pos_tensor,f"{output_case_path}/geom_pos_tensors.pt")
                geom_area_tensor = torch.tensor(geom_areas_normalized, dtype=torch.float32)
                torch.save(geom_area_tensor,f"{output_case_path}/geom_area_tensors.pt")


            # Delete RAW FILE AFTER
            # Delete File After Processed
            os.remove(f"{local_path}")
    
        except HfHubHTTPError as e:
            print(f"Failed to download {filename}: {e}\n")
            # You can add retry logic here if needed
    
    # Deletes File 
    shutil.rmtree(f"temp_raw_drivAerML/{run}")
