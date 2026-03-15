import os
import shutil
from tqdm import tqdm

## Function: downsample images in one folder
def downsample_images(input_folder, output_folder, factor=3):
    os.makedirs(output_folder, exist_ok=True) #creates output folder

    #collect only x component images
    x_images = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith("_x.jpg")
    ])

    #downsample by selecting every 3rd frame
    for index, x_img in enumerate(x_images):

        if index%factor == 0:
            y_img = x_img.replace("_x.jpg","_y.jpg")

            x_input_path = os.path.join(input_folder, x_img)
            y_input_path = os.path.join(input_folder, y_img)

            x_output_path = os.path.join(output_folder, x_img)
            y_output_path = os.path.join(output_folder, y_img)

            if os.path.exists(y_input_path):
                shutil.copy2(x_input_path, x_output_path)
                shutil.copy2(y_input_path, y_output_path)

            else:
                print(f"<<WARNING>> Missing pair for {x_img}")

## Function: traverse entire optimal flow folder
def traverse_optimal(input_root, output_root, factor=3):
    for root, dirs, files in os.walk(input_root):
        flow_images = [f for f in files if f.endswith("_x.jpg")]

        if len(flow_images) == 0:
            continue

        relative_path = os.path.relpath(root, input_root)
        output_folder = os.path.join(output_root, relative_path)

        print(f"\nProcessing folder: {root}")

        downsample_images(root, output_folder, factor)

## Execution
if __name__=="__main__":
    INPUT_FLOW = r"D:\MMASD_project\optical_flow"
    OUTPUT_FLOW = r"D:\MMASD_dataset\optimal_flow_downsampled"

    DOWNSAMPLE_FACTOR = 3

    print("\nStarting Downsampling...\n")

    traverse_optimal(
        INPUT_FLOW,
        OUTPUT_FLOW,
        DOWNSAMPLE_FACTOR
    )

    print("\n<<COMPLETED>>\n")