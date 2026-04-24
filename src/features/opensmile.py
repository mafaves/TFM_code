import opensmile
import pandas as pd
from tqdm import tqdm
import numpy as np


def extract_opensmile_features(
    audio_chunks,
    labels,
    patient_ids,
    exercises=None,
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    sampling_rate=16000,
    verbose=True
):
    """
    Extract OpenSMILE features from audio chunks.

    Args:
        audio_chunks (np.array): Array of audio waveforms.
        labels (np.array): Labels for each chunk.
        patient_ids (np.array): Patient IDs for each chunk.
        exercises (np.array, optional): Exercise names for each chunk.
        feature_set (opensmile.FeatureSet): OpenSMILE feature set.
        feature_level (opensmile.FeatureLevel): OpenSMILE feature level.
        sampling_rate (int): Sampling rate.
        verbose (bool): Show progress bar.

    Returns:
        pd.DataFrame: DataFrame with extracted features + metadata columns.
            Columns: patient_id, label, exercise, [feature columns...]
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    all_features = []
    iterator = tqdm(enumerate(audio_chunks), total=len(audio_chunks)) if verbose else enumerate(audio_chunks)

    for i, chunk in iterator:
        features = smile.process_signal(chunk, sampling_rate=sampling_rate)

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        all_features.append(features)

    df = pd.concat(all_features, ignore_index=True)

    # Add metadata columns
    df.insert(0, 'patient_id', patient_ids)
    df.insert(1, 'label', labels)
    
    # Add exercise column if provided
    if exercises is not None:
        if len(exercises) != len(audio_chunks):
            raise ValueError(f"Exercises length ({len(exercises)}) doesn't match chunks ({len(audio_chunks)})")
        df.insert(2, 'exercise', exercises)
    else:
        df.insert(2, 'exercise', 'unknown')

    nan_cols = df.isnull().sum()
    cols_with_nan = nan_cols[nan_cols > 0].index.tolist()
    if cols_with_nan:
        print(f"Columns with NaN values: {cols_with_nan}")
        df = df.drop(columns=cols_with_nan)

    print(f"OpenSMILE features shape: {df.shape}")
    print(f"Unique exercises: {df['exercise'].unique()}")

    return df