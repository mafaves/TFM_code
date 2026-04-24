import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_librosa_features(
    audio_chunks,
    labels,
    patient_ids,
    exercises=None,
    sr=16000,
    n_mfcc=13,
    n_mels=128,
    hop_length=512,
    n_fft=2048,
    verbose=True
):
    """
    Extract acoustic features using Librosa.

    Features extracted:
        - MFCCs: mean, std, max, min (per coefficient)
        - Spectral features: centroid, bandwidth, rolloff, contrast, flatness
        - Rhythm: zero crossing rate, RMS energy

    Args:
        audio_chunks (np.array): Array of audio waveforms.
        labels (np.array): Labels for each chunk.
        patient_ids (np.array): Patient IDs for each chunk.
        exercises (np.array, optional): Exercise names for each chunk.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs to extract.
        n_mels (int): Number of mel bands.
        hop_length (int): Hop length for STFT.
        n_fft (int): FFT window size.
        verbose (bool): Show progress bar.

    Returns:
        pd.DataFrame: DataFrame with extracted features + metadata columns.
            Columns: patient_id, label, exercise, [feature columns...]
    """
    features_list = []
    iterator = tqdm(enumerate(audio_chunks), total=len(audio_chunks)) if verbose else enumerate(audio_chunks)

    for i, y in iterator:
        try:
            features = {}

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
            for j, mfcc in enumerate(mfccs):
                features[f'mfcc_{j}_mean'] = np.mean(mfcc)
                features[f'mfcc_{j}_std'] = np.std(mfcc)

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=mel_spec))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=mel_spec))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=mel_spec))
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(S=mel_spec))

            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['rms'] = np.mean(librosa.feature.rms(y=y))

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
            for j, c in enumerate(chroma):
                features[f'chroma_{j}'] = np.mean(c)

            features_list.append(features)

        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            features_list.append({k: np.nan for k in features_list[0].keys()}) if features_list else None

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

    print(f"Librosa features shape: {df.shape}")
    print(f"Unique exercises: {df['exercise'].unique()}")

    return df