import os #Handles folder navigation
import shutil #used to copy files safely 
from tqdm import tqdm #(optional) shows progress bar

## function: downsample JSON files in one folder
def downsample_json(input_folder, output_folder, factor=3):
    os.makedirs(output_folder, exist_ok=True) #create output folder if it doesn't exist

    #Get all json files and sort them
    files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".json")
    ])

    #Loop through files and copy selected ones
    for index, file in enumerate(files):
        #keep every 3rd file
        if index%factor==0:
            src = os.path.join(input_folder, file)
            dst = os.path.join(output_folder, file)

            #copy files instead of moving or deleting #preserves timestamp
            shutil.copy2(src,dst) 

## Function: Traverse entire dataset (handles subfolders)
def traverse(input_root, output_root, factor=3):
    for root, dirs, files in os.walk(input_root): #os.walk() automatically goes through main folder->subfolder->files
        #check if current folder contains json files
        json_files = [f for f in files if f.endswith(".json")]

        if len(json_files)>0: #if folder contains JSON, downsample it
            #construct equivalent output folder path
            relative_path = os.path.relpath(root, input_root) #preserve original hierarchy
            output_folder = os.path.join(output_root, relative_path)

            print(f"\nProcessing folder: {root}")

            downsample_json(root, output_folder, factor)

## Execution
if __name__=="__main__":
    INPUT_DATASET = r"D:\MMASD_project\2D skeleton\2D_openpose"
    OUTPUT_DATASET = r"D:\MMASD_dataset\2D_openpose_downsampled"

    DOWNSAMPLE_FACTOR = 3

    print("\nStarting Downsampling...\n")

    traverse(
        INPUT_DATASET,
        OUTPUT_DATASET,
        DOWNSAMPLE_FACTOR
    )

    print("<<COMPLETE>>")