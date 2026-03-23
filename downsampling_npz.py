import os #folder navigation
import numpy as np #loads .npz file and processes arrays
from tqdm import tqdm #shows progress bar 

## Function: downsample one npz file
def downsample_npz(input_file, output_file, factor=3):
    #load npz file
    data = np.load(input_file, allow_pickle=True)

    #temporary dictionary to store processed arrays
    processed_data = {}

    #loop through all arrays stored inside npz 
    for key in data.files:
        arr = data[key]
        if isinstance(arr,np.ndarray) and arr.shape[0]>1: #shape[0] represents num_frames
            #keep every 3rd frame
            processed_data[key] = arr[::factor]

        else:
            #copy non-frame data unchanged
            processed_data[key] = arr
    
    np.savez_compressed(output_file, **processed_data)

## Function: traverse dataset folders (handles subfolders)

def traverse_romp(input_root, output_root, factor=3):
    for root, dirs, files in os.walk(input_root):
        #identify npz files inside current folder
        npz_files = [f for f in files if f.endswith(".npz")]

        if len(npz_files) == 0:
            continue

        #create equivalent output folder path
        relative_path = os.path.relpath(root, input_root)  #ensures output dataset looks identical
        output_folder = os.path.join(output_root, relative_path)

        os.makedirs(output_folder, exist_ok=True)

        print(f"\nProcessing Folder: {root}")

        #process each npz file
        for file in tqdm(npz_files):
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_folder, file)

            downsample_npz(input_file, output_file, factor)

## Execution
if __name__=="__main__":

    #romp 2D 
    ROMP_2D_INPUT = r"D:\MMASD_project\2D skeleton\ROMP_2D_Coordinates"
    ROMP_2D_OUTPUT = r"D:\MMASD_project\2d_romp_downsampled"

    #romp 3d 
    ROMP_3D_INPUT = r"D:\MMASD_project\3D skeleton-20260206T170206Z-1-001\ROMP_3D_Coordinates"
    ROMP_3D_OUTPUT = r"D:\MMASD_project\3d_romp_downsampled"

    #romp 3d 71 joint
    ROMP_3D_71_INPUT = r"D:\MMASD_project\3D skeleton-20260206T170206Z-1-001\ROMP_3D_Coordinates_71-joints"
    ROMP_3D_71_OUTPUT = r"D:\MMASD_project\3d_71joints_downsampled"

    DOWNSAMPLE_FACTOR = 3

    print("\nStarting Downsampling...\n")

    #process each folder

    traverse_romp(ROMP_2D_INPUT, ROMP_2D_OUTPUT, DOWNSAMPLE_FACTOR)
    traverse_romp(ROMP_3D_INPUT, ROMP_3D_OUTPUT, DOWNSAMPLE_FACTOR)
    traverse_romp(ROMP_3D_71_INPUT, ROMP_3D_71_OUTPUT, DOWNSAMPLE_FACTOR)

    print("<<ALL DONE>>")