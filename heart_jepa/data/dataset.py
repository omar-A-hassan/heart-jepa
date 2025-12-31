"""Dataset classes for PCG data loading."""

from typing import Optional, Callable, List, Tuple, Union
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .preprocessing import (
    load_pcg,
    preprocess_pcg,
    process_pcg_to_spectrogram,
    spectrogram_to_tensor,
    segment_pcg,
    SPECTROGRAM_CONFIG,
)
from .augmentations import MultiViewTransform, TestTransform


class PCGDataset(Dataset):
    """
    Generic PCG dataset for loading audio files.

    Supports:
        - Loading from directory of audio files
        - Optional labels from CSV
        - Multi-view augmentation for LEJEPA training
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        sr: int = 2000,
        duration: float = 5.0,
        split: str = "train",
        config: Optional[dict] = None,
        file_extension: str = ".wav",
    ):
        """
        Args:
            data_dir: Directory containing audio files
            labels_file: Optional CSV with columns [filename, label]
            transform: Transform to apply (MultiViewTransform or TestTransform)
            sr: Target sample rate
            duration: Duration of each sample in seconds
            split: Dataset split ("train", "val", "test")
            config: Spectrogram configuration
            file_extension: Audio file extension
        """
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.duration = duration
        self.config = config or SPECTROGRAM_CONFIG
        self.split = split

        # Set up transform
        if transform is None:
            if split == "train":
                self.transform = MultiViewTransform(n_views=4, config=self.config)
            else:
                self.transform = TestTransform(config=self.config)
        else:
            self.transform = transform

        # Load file list
        self.files = sorted(self.data_dir.glob(f"*{file_extension}"))

        # Load labels if provided
        self.labels = None
        self.label_map = None
        if labels_file is not None:
            self._load_labels(labels_file)

    def _load_labels(self, labels_file: Union[str, Path]):
        """Load labels from CSV file."""
        df = pd.read_csv(labels_file)

        # Assume columns: filename, label
        self.labels = {}
        unique_labels = sorted(df['label'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        for _, row in df.iterrows():
            filename = row['filename']
            label = self.label_map[row['label']]
            self.labels[filename] = label

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Returns:
            Tuple of (spectrogram tensor, label)
            - For train: spectrogram shape (n_views, 1, H, W)
            - For val/test: spectrogram shape (1, H, W)
            - Label is -1 if no labels provided
        """
        file_path = self.files[idx]

        # Load and preprocess PCG
        pcg, _ = load_pcg(file_path, target_sr=self.sr, duration=self.duration)

        # Pad or truncate to exact duration
        target_samples = int(self.duration * self.sr)
        if len(pcg) < target_samples:
            pcg = np.pad(pcg, (0, target_samples - len(pcg)), mode='constant')
        else:
            pcg = pcg[:target_samples]

        # Apply transform
        spec = self.transform(pcg, self.sr)

        # Get label
        label = -1
        if self.labels is not None:
            filename = file_path.name
            label = self.labels.get(filename, -1)

        return spec, label

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        if self.label_map is not None:
            return len(self.label_map)
        return 0


class PhysioNetDataset(Dataset):
    """
    Dataset for PhysioNet/CinC Challenge 2016 heart sound data.

    Downloads and processes data from PhysioNet.
    Labels: 0 = Normal, 1 = Abnormal
    """

    PHYSIONET_URL = "https://physionet.org/files/challenge-2016/1.0.0/"

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        sr: int = 2000,
        duration: float = 5.0,
        config: Optional[dict] = None,
        download: bool = False,
    ):
        """
        Args:
            data_dir: Directory to store/load data
            split: Dataset split ("train", "val", "test")
            transform: Transform to apply
            sr: Target sample rate
            duration: Duration per sample
            config: Spectrogram configuration
            download: Whether to download data if not present
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sr = sr
        self.duration = duration
        self.config = config or SPECTROGRAM_CONFIG

        # Set up transform
        if transform is None:
            if split == "train":
                self.transform = MultiViewTransform(n_views=4, config=self.config)
            else:
                self.transform = TestTransform(config=self.config)
        else:
            self.transform = transform

        # Download if needed
        if download and not self._check_exists():
            self._download()

        # Load data
        self.samples, self.labels = self._load_data()

    def _check_exists(self) -> bool:
        """Check if data exists."""
        return (self.data_dir / "training-a").exists()

    def _download(self):
        """Download PhysioNet data using wfdb."""
        import wfdb

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Download training sets a-f
        for subset in ['a', 'b', 'c', 'd', 'e', 'f']:
            db_name = f"challenge-2016/training-{subset}"
            output_dir = self.data_dir / f"training-{subset}"

            if not output_dir.exists():
                print(f"Downloading {db_name}...")
                try:
                    wfdb.dl_database(db_name, str(output_dir))
                except Exception as e:
                    print(f"Error downloading {db_name}: {e}")
                    print("You may need to download manually from PhysioNet")

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        """Load file paths and labels."""
        samples = []
        labels = []

        # Load from each training subset
        for subset_dir in sorted(self.data_dir.glob("training-*")):
            # Read REFERENCE.csv for labels
            ref_file = subset_dir / "REFERENCE.csv"
            if not ref_file.exists():
                # Try alternative name
                ref_file = subset_dir / "REFERENCE-original.csv"

            if ref_file.exists():
                df = pd.read_csv(ref_file, header=None, names=['record', 'label'])

                for _, row in df.iterrows():
                    record_name = row['record']
                    label = 1 if row['label'] == -1 else 0  # -1 = abnormal, 1 = normal

                    # Find wav file
                    wav_file = subset_dir / f"{record_name}.wav"
                    if wav_file.exists():
                        samples.append(wav_file)
                        labels.append(label)

        # Split data (80% train, 10% val, 10% test)
        total = len(samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)

        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(total)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]

        samples = [samples[i] for i in split_indices]
        labels = [labels[i] for i in split_indices]

        return samples, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        file_path = self.samples[idx]
        label = self.labels[idx]

        # Load PCG
        pcg, _ = load_pcg(file_path, target_sr=self.sr, duration=self.duration)

        # Pad or truncate
        target_samples = int(self.duration * self.sr)
        if len(pcg) < target_samples:
            pcg = np.pad(pcg, (0, target_samples - len(pcg)), mode='constant')
        else:
            pcg = pcg[:target_samples]

        # Apply transform
        spec = self.transform(pcg, self.sr)

        return spec, label

    @property
    def num_classes(self) -> int:
        return 2  # Normal / Abnormal


class SegmentationDataset(Dataset):
    """
    Dataset for PCG segmentation with pseudo-labels.

    Wraps any PCG dataset and generates frame-level pseudo-labels
    for S1, S2, Systole, Diastole segmentation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        sr: int = 2000,
        duration: float = 5.0,
        output_frames: int = 224,
        config: Optional[dict] = None,
    ):
        """
        Args:
            data_dir: Directory containing audio files
            split: Dataset split
            transform: Transform to apply (should be TestTransform for single view)
            sr: Target sample rate
            duration: Duration per sample
            output_frames: Number of output frames for labels
            config: Spectrogram configuration
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sr = sr
        self.duration = duration
        self.output_frames = output_frames
        self.config = config or SPECTROGRAM_CONFIG

        if transform is None:
            self.transform = TestTransform(config=self.config)
        else:
            self.transform = transform

        # Load file list from PhysioNet structure
        self.samples = self._load_samples()

        # Import pseudo-label generator
        from ..utils.pseudo_labels import generate_pseudo_labels, validate_pseudo_labels
        self.generate_pseudo_labels = generate_pseudo_labels
        self.validate_pseudo_labels = validate_pseudo_labels

    def _load_samples(self) -> List[Path]:
        """Load sample file paths."""
        samples = []
        for subset_dir in sorted(self.data_dir.glob("training-*")):
            for wav_file in subset_dir.glob("*.wav"):
                samples.append(wav_file)

        # Split data
        total = len(samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)

        np.random.seed(42)
        indices = np.random.permutation(total)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        else:
            split_indices = indices[val_end:]

        return [samples[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item with spectrogram and pseudo-labels.

        Returns:
            Tuple of (spectrogram, labels)
            - spectrogram: (1, H, W)
            - labels: (output_frames,) with class indices
        """
        file_path = self.samples[idx]

        # Load PCG
        pcg, _ = load_pcg(file_path, target_sr=self.sr, duration=self.duration)

        # Pad or truncate
        target_samples = int(self.duration * self.sr)
        if len(pcg) < target_samples:
            pcg = np.pad(pcg, (0, target_samples - len(pcg)), mode='constant')
        else:
            pcg = pcg[:target_samples]

        # Preprocess for pseudo-label generation
        pcg_clean = preprocess_pcg(pcg, self.sr)

        # Generate pseudo-labels
        labels = self.generate_pseudo_labels(
            pcg_clean,
            sr=self.sr,
            output_frames=self.output_frames,
        )

        # Apply transform to get spectrogram
        spec = self.transform(pcg, self.sr)

        return spec, torch.from_numpy(labels).long()

    @property
    def num_classes(self) -> int:
        return 7  # background, S1, systole, S2, diastole, S3, S4


class CirCorDataset(Dataset):
    """
    Dataset for CirCor DigiScope Phonocardiogram Dataset.

    Labels: Murmur present/absent/unknown
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        sr: int = 2000,
        duration: float = 5.0,
        config: Optional[dict] = None,
        include_unknown: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing CirCor data
            split: Dataset split
            transform: Transform to apply
            sr: Target sample rate
            duration: Duration per sample
            config: Spectrogram configuration
            include_unknown: Whether to include 'unknown' murmur label
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sr = sr
        self.duration = duration
        self.config = config or SPECTROGRAM_CONFIG
        self.include_unknown = include_unknown

        if transform is None:
            if split == "train":
                self.transform = MultiViewTransform(n_views=4, config=self.config)
            else:
                self.transform = TestTransform(config=self.config)
        else:
            self.transform = transform

        self.samples, self.labels = self._load_data()

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        """Load CirCor data."""
        samples = []
        labels = []

        # Load patient data
        csv_file = self.data_dir / "training_data.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)

            # Label mapping
            label_map = {"Absent": 0, "Present": 1}
            if self.include_unknown:
                label_map["Unknown"] = 2

            for _, row in df.iterrows():
                patient_id = row['Patient ID']
                murmur = row.get('Murmur', 'Unknown')

                if murmur not in label_map:
                    continue

                # Find all recordings for this patient
                for wav_file in self.data_dir.glob(f"{patient_id}*.wav"):
                    samples.append(wav_file)
                    labels.append(label_map[murmur])

        # Split data
        total = len(samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)

        np.random.seed(42)
        indices = np.random.permutation(total)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        else:
            split_indices = indices[val_end:]

        samples = [samples[i] for i in split_indices]
        labels = [labels[i] for i in split_indices]

        return samples, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        file_path = self.samples[idx]
        label = self.labels[idx]

        pcg, _ = load_pcg(file_path, target_sr=self.sr, duration=self.duration)

        target_samples = int(self.duration * self.sr)
        if len(pcg) < target_samples:
            pcg = np.pad(pcg, (0, target_samples - len(pcg)), mode='constant')
        else:
            pcg = pcg[:target_samples]

        spec = self.transform(pcg, self.sr)
        return spec, label

    @property
    def num_classes(self) -> int:
        return 3 if self.include_unknown else 2
