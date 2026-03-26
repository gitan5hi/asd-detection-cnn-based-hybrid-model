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

    def __init__(self, sequence_length=10):

        self.sequence_length = sequence_length

        self.optimal_flow_dir = r"D:\MMASD_dataset\optimal_flow_downsampled"
        self.openpose_dir = r"D:\MMASD_dataset\2D_openpose_downsampled"
        self.romp3d_dir = r"D:\MMASD_dataset\3d_romp_downsampled"
        self.clinical_csv = r"D:\MMASD_dataset\ADOS_rating.csv"

        # =====================
        # LOAD CSV + FIX COLUMN
        # =====================
        df = pd.read_csv(self.clinical_csv)

        df = df.rename(columns={
            "ADOS Comparison Score (1-10) <5  not very autistic. ASD people usually fall 5-10. 8-10=Severe, 5-7=moderate, 1-4=mild": "score"
        })

        df = df.drop_duplicates(subset="ID#", keep="first")
        df["ID#"] = df["ID#"].astype(str)

        self.clinical_lookup = df.set_index("ID#").to_dict("index")

        # =====================
        # BUILD SAMPLES
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
                row = self.clinical_lookup.get(subject_id, None)

                if row is None:
                    continue

                score = row.get("score", None)

                # Skip invalid
                if pd.isna(score):
                    continue

                score = int(score)

                # Keep only moderate + severe
                if score < 5:
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
    # UTIL
    # =====================
    def extract_subject_id(self, folder_name):
        match = re.search(r"\d+", folder_name)
        return match.group(0) if match else None

    def pad_sequence(self, data):
        if len(data) == 0:
            raise ValueError("Empty sequence encountered")

        if len(data) < self.sequence_length:
            pad = [data[-1]] * (self.sequence_length - len(data))
            data.extend(pad)

        return data[:self.sequence_length]

    # =====================
    # OPTICAL FLOW
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

        frames = self.pad_sequence(frames)

        return torch.stack(frames)

    # =====================
    # 2D SKELETON
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

        skeleton = self.pad_sequence(skeleton)
        skeleton = np.array(skeleton)

        return torch.tensor(skeleton, dtype=torch.float32)

    # =====================
    # 3D SKELETON
    # =====================
    def load_romp3d(self, pose, subject):

        subject_path = os.path.join(self.romp3d_dir, pose, subject)

        npz_files = sorted(os.listdir(subject_path))
        skeleton = []

        for i in range(min(self.sequence_length, len(npz_files))):

            data = np.load(os.path.join(subject_path, npz_files[i]))
            joints = data["coordinates"]

            if joints.ndim == 3:
                joints = joints[0]
            if joints.ndim == 4:
                joints = joints[0][0]

            skeleton.append(joints)

        skeleton = self.pad_sequence(skeleton)
        skeleton = np.stack(skeleton)

        return torch.tensor(skeleton, dtype=torch.float32)

    # =====================
    # LABEL (BINARY: MODERATE vs SEVERE)
    # =====================
    def load_clinical(self, subject_id):

        row = self.clinical_lookup[subject_id]
        score = int(row["score"])

        if score <= 7:
            label = 0   # Moderate
        else:
            label = 1   # Severe

        return torch.tensor(label, dtype=torch.long)

    # =====================
    # GET ITEM
    # =====================
    def __getitem__(self, idx):

        pose, subject_folder, subject_id = self.samples[idx]

        return {
            "optimal_flow": self.load_optical_flow(pose, subject_folder),
            "skeleton2d": self.load_openpose(pose, subject_folder),
            "skeleton3d": self.load_romp3d(pose, subject_folder),
            "label": self.load_clinical(subject_id)
        }

    def __len__(self):
        return len(self.samples)