import os
import pandas as pd

LABEL_MAPPING = {
    'HC': 0,  # Healthy Control
    'NFC': 0, # Negative Family Carrier (considered as healthy for binary classification)
    'AC': 1,  # Asymtomatic Carrier of G2019S LRRK2 mutation
    'PD': 2   # Parkinson's Disease
}


def load_audio_data(
    root_directory,
    start_with=None,
    exact_name=None,
    patient_type_mapping=LABEL_MAPPING,
    audio_extensions=['.wav']
):
    """
    Loads audio file paths and labels from the HUMV directory structure.

    Directory structure expected:
        root/
        ├── HC/
        │   └── HUMV_HC_001/
        │       └── vocal.wav
        ├── NFC/
        │   └── HUMV_NFC_001/
        │       └── vocal.wav
        ├── AC/
        │   └── HUMV_AC_001/
        │       └── vocal.wav
        └── PD/
            └── HUMV_PD_001/
                └── vocal.wav

    Args:
        root_directory (str): Root directory containing patient type folders (HC, NFC, AC, PD).
        start_with (str, optional): Filter files starting with this prefix (case-insensitive).
        exact_name (str, optional): Filter files that exactly match this name.
        patient_type_mapping (dict): Mapping from patient types to numerical labels.
        audio_extensions (list): Valid audio file extensions (e.g., ['.wav', '.mp3']).

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'Patient': Patient ID (e.g., 'HUMV_PD_001')
            - 'Label': Numeric label (0=HC/NFC, 1=AC, 2=PD)
            - 'File_Path': Full path to audio file
            - 'Audio_Name': Audio file name without extension
    """
    data = []

    for patient_type in patient_type_mapping.keys():
        category_dir = os.path.join(root_directory, patient_type)
        if not os.path.isdir(category_dir):
            continue

        for patient in os.listdir(category_dir):
            patient_dir = os.path.join(category_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            if len(patient.split("_")) != 3:
                continue

            _, ptype, patient_number = patient.split("_")

            for file in os.listdir(patient_dir):
                if not file.endswith(tuple(audio_extensions)):
                    continue

                if start_with and not file.lower().startswith(start_with.lower()):
                    continue
                if exact_name and file != exact_name:
                    continue

                audio_path = os.path.join(patient_dir, file)
                audio_name = os.path.splitext(file)[0]

                patient_id = f"HUMV_{ptype}_{patient_number}"

                data.append({
                    'Patient': patient_id,
                    'Label': patient_type_mapping[ptype],
                    'File_Path': audio_path,
                    'Audio_Name': audio_name
                })

    df = pd.DataFrame(data)

    if df.empty:
        print("No audio files found.")
    else:
        print(f"Loaded {len(df)} audio files.")
        print("Label distribution:")
        print(df['Label'].value_counts().sort_index())

    return df


def filter_binary(df, labels_to_keep):
    """
    Filter DataFrame to keep only specified labels and remap them to binary (0/1).

    Args:
        df (pd.DataFrame): DataFrame with 'Label' column.
        labels_to_keep (list): Labels to keep (e.g., [0, 2] for HC vs PD).

    Returns:
        pd.DataFrame: Filtered DataFrame with remapped labels (0, 1).
    """
    df_filtered = df[df['Label'].isin(labels_to_keep)].copy()

    label_map = {labels_to_keep[0]: 0, labels_to_keep[1]: 1}
    df_filtered['Label'] = df_filtered['Label'].map(label_map)

    print(f"Filtered DataFrame shape: {df_filtered.shape}")
    print(f"Label distribution after filtering:")
    print(df_filtered['Label'].value_counts().sort_index())

    return df_filtered