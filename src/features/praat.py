import parselmouth
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_praat_features(
    audio_chunks,
    labels,
    patient_ids,
    exercises=None,
    sampling_rate=16000,
    verbose=True
):
    """
    Extract acoustic features using Praat (via Parselmouth).

    Features extracted:
        - Pitch: mean, min, max, std
        - Formants: F1, F2, F3 mean
        - Intensity: mean, min, max, std
        - Jitter, Shimmer
        - HNR (Harmonics-to-Noise Ratio)

    Args:
        audio_chunks (np.array): Array of audio waveforms.
        labels (np.array): Labels for each chunk.
        patient_ids (np.array): Patient IDs for each chunk.
        exercises (np.array, optional): Exercise names for each chunk.
        sampling_rate (int): Sampling rate.
        verbose (bool): Show progress bar.

    Returns:
        pd.DataFrame: DataFrame with extracted features + metadata columns.
            Columns: patient_id, label, exercise, [feature columns...]
    """
    features_list = []
    iterator = tqdm(enumerate(audio_chunks), total=len(audio_chunks)) if verbose else enumerate(audio_chunks)

    for i, chunk in iterator:
        try:
            sound = parselmouth.Sound(chunk.astype("float64"), sampling_frequency=sampling_rate)

            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]

            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else np.nan
            pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else np.nan
            pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else np.nan
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else np.nan

            try:
                formant = parselmouth.praat.call(sound, "To Formant (burg)", 0.025, 5, 5500, 0.025, 50)
                f1_mean = parselmouth.praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
                f2_mean = parselmouth.praat.call(formant, "Get mean", 2, 0, 0, "Hertz")
                f3_mean = parselmouth.praat.call(formant, "Get mean", 3, 0, 0, "Hertz")
            except:
                f1_mean = f2_mean = f3_mean = np.nan

            intensity = sound.to_intensity()
            intensity_values = intensity.values[0]
            intensity_mean = np.mean(intensity_values)
            intensity_min = np.min(intensity_values)
            intensity_max = np.max(intensity_values)
            intensity_std = np.std(intensity_values)

            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            hnr = sound.to_harmonicity_cc()
            hnr_values = hnr.values[0]
            hnr_mean = np.mean(hnr_values)

            features = {
                'pitch_mean': pitch_mean,
                'pitch_min': pitch_min,
                'pitch_max': pitch_max,
                'pitch_std': pitch_std,
                'f1_mean': f1_mean,
                'f2_mean': f2_mean,
                'f3_mean': f3_mean,
                'intensity_mean': intensity_mean,
                'intensity_min': intensity_min,
                'intensity_max': intensity_max,
                'intensity_std': intensity_std,
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr_mean': hnr_mean
            }
            features_list.append(features)

        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            features_list.append({
                'pitch_mean': np.nan,
                'pitch_min': np.nan,
                'pitch_max': np.nan,
                'pitch_std': np.nan,
                'f1_mean': np.nan,
                'f2_mean': np.nan,
                'f3_mean': np.nan,
                'intensity_mean': np.nan,
                'intensity_min': np.nan,
                'intensity_max': np.nan,
                'intensity_std': np.nan,
                'jitter': np.nan,
                'shimmer': np.nan,
                'hnr_mean': np.nan
            })

    df = pd.DataFrame(features_list)
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

    print(f"Praat features shape: {df.shape}")
    print(f"Unique exercises: {df['exercise'].unique()}")

    return df