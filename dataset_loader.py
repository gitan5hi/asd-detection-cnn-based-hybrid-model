import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class MMASDDataset(Dataset):

    def __init__(self, sequence_length=30):

        self.sequence_length = sequence_length

        self.optimal_flow_dir = r"D:\MMASD_dataset\optimal_flow_downsampled"
        self.openpose_dir = r"D:\MMASD_dataset\2D_openpose_downsampled"
        self.romp3d_dir = r"D:\MMASD_dataset\3d_romp_downsampled"
        self.clinical_csv = r"D:\MMASD_dataset\ADOS_rating.csv"

        # =====================
        # Clinical Lookup
        # =====================
        df = pd.read_csv(self.clinical_csv)

        df = df.drop_duplicates(subset="ID#", keep="first")
        df["ID#"] = df["ID#"].astype(str)

        self.clinical_lookup = df.set_index("ID#").to_dict("index")

        # =====================
        # Build Sample Index
        # =====================
        self.samples = []

        poses = os.listdir(self.optimal_flow_dir)

        for pose in poses:

            flow_pose = os.path.join(self.optimal_flow_dir, pose)
            pose2d_pose = os.path.join(self.openpose_dir, pose)
            pose3d_pose = os.path.join(self.romp3d_dir, pose)

            if not (os.path.exists(flow_pose) and os.path.exists(pose2d_pose) and os.path.exists(pose3d_pose)):
                continue

            for subject_folder in os.listdir(flow_pose):

                subject_id = self.extract_subject_id(subject_folder)

                if subject_id not in self.clinical_lookup:
                    continue

                flow_subject = os.path.join(flow_pose, subject_folder)
                pose2d_subject = os.path.join(pose2d_pose, subject_folder)
                pose3d_subject = os.path.join(pose3d_pose, subject_folder)

                if not (os.path.exists(pose2d_subject) and os.path.exists(pose3d_subject)):
                    continue

                if len(os.listdir(flow_subject)) == 0:
                    continue

                self.samples.append((pose, subject_folder, subject_id))

        print(f"Loaded {len(self.samples)} valid samples")

    # =====================
    # Extract numeric ID
    # =====================
    def extract_subject_id(self, folder_name):

        match = re.search(r"\d+", folder_name)
        return match.group(0) if match else None

    # =====================
    # Optical Flow
    # =====================
    def load_optical_flow(self, pose, subject):

        subject_path = os.path.join(self.optimal_flow_dir, pose, subject)

        x_files = sorted([f for f in os.listdir(subject_path) if "_x.jpg" in f])
        y_files = sorted([f for f in os.listdir(subject_path) if "_y.jpg" in f])

        frame_count = min(self.sequence_length, len(x_files), len(y_files))

        frames = []

        for i in range(frame_count):
            x = Image.open(os.path.join(subject_path, x_files[i])).convert("L")
            y = Image.open(os.path.join(subject_path, y_files[i])).convert("L")

            x = image_transform(x)
            y = image_transform(y)

            frames.append(torch.cat([x, y], dim=0))

        if len(frames) == 0:
            raise ValueError("No optical flow frames")

        return torch.stack(frames)

    # =====================
    # 2D Skeleton
    # =====================
    def load_openpose(self, pose, subject):

        subject_path = os.path.join(self.openpose_dir, pose, subject)

        json_files = sorted(os.listdir(subject_path))
        skeleton = []

        for i in range(min(self.sequence_length, len(json_files))):

            with open(os.path.join(subject_path, json_files[i])) as f:
                data = json.load(f)

            if len(data["people"]) > 0:
                kp = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(25, 3)
            else:
                kp = np.zeros((25, 3))

            skeleton.append(kp)

        return torch.tensor(skeleton, dtype=torch.float32)

    # =====================
    # 3D Skeleton
    # =====================
    def load_romp3d(self, pose, subject):

        subject_path = os.path.join(self.romp3d_dir, pose, subject)

        npz_files = sorted(os.listdir(subject_path))
        skeleton = []

        for i in range(min(self.sequence_length, len(npz_files))):

            file_path = os.path.join(subject_path, npz_files[i])
            data = np.load(file_path)

            joints = data["coordinates"]

            joints = np.array(joints)

            if joints.ndim == 3:
                joints = joints[0]

            if joints.ndim == 4:
                joints = joints[0][0]

            skeleton.append(joints)

        if len(skeleton) < self.sequence_length:
            pad = [skeleton[-1]] * (self.sequence_length - len(skeleton))
            skeleton.extend(pad)

        skeleton = np.stack(skeleton)

        return torch.tensor(skeleton, dtype=torch.float32)

    # =====================
    # Clinical
    # =====================
    def convert_age(self, age_str):

        years = 0
        months = 0

        if "Y" in age_str:
            years = int(age_str.split("Y")[0])

        if "M" in age_str:
            months = int(age_str.split(",")[1].replace("M", ""))

        return years + months / 12

    def load_clinical(self, subject_id):

        row = self.clinical_lookup[subject_id]

        label = int(row["ADOS-2 classification/Dx"])
        gender = 1 if row["Gender"] == "M" else 0
        age = self.convert_age(row["Chronological Age"])

        meta = torch.tensor([age, gender], dtype=torch.float32)

        return torch.tensor(label), meta

    # =====================
    # Main Getter
    # =====================
    def __getitem__(self, idx):

        pose, subject_folder, subject_id = self.samples[idx]

        optimal_flow = self.load_optical_flow(pose, subject_folder)
        skeleton2d = self.load_openpose(pose, subject_folder)
        skeleton3d = self.load_romp3d(pose, subject_folder)
        label, meta = self.load_clinical(subject_id)

        return {
            "optimal_flow": optimal_flow,
            "skeleton2d": skeleton2d,
            "skeleton3d": skeleton3d,
            "meta": meta,
            "label": label
        }

    def __len__(self):
        return len(self.samples)