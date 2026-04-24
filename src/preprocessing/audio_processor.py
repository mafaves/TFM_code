import librosa
import numpy as np


def process_and_split_audio(
    audio_path,
    label,
    patient_id,
    exercise=None,
    start_time=0,
    chunk_duration=5,
    max_duration=None,
    target_sr=16000,
    remove_silence=True,
    top_db=25,
    silence_duration=0.5,
    overlap=0,
    min_chunk_ratio=0.7
):
    """
    Loads audio file, processes it, and splits into chunks with optional overlap.

    Args:
        audio_path (str): Path to the audio file.
        label (int): Label associated with the audio.
        patient_id (str): Patient ID.
        exercise (str, optional): Exercise name (e.g., 'vocal', 'text').
        start_time (float): Start time in seconds.
        chunk_duration (float): Duration of each chunk in seconds.
        max_duration (float, optional): Maximum duration to process. None = use entire audio.
        target_sr (int): Target sample rate.
        remove_silence (bool): Whether to remove silence.
        top_db (float): Threshold for silence detection.
        silence_duration (float): Max silence duration to keep (seconds).
        overlap (float): Overlap between chunks in seconds.
        min_chunk_ratio (float): Minimum chunk length ratio (0.7 = keep 70% of target).

    Returns:
        list: List of tuples (audio_chunk, label, patient_id, exercise).
    """
    y, sr = librosa.load(audio_path, sr=None)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    y = librosa.util.normalize(y)

    if remove_silence:
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)
        processed_audio = []
        max_silence_samples = int(silence_duration * target_sr)

        for i, (start, end) in enumerate(non_silent_intervals):
            processed_audio.append(y[start:end])
            if i < len(non_silent_intervals) - 1:
                next_start = non_silent_intervals[i + 1][0]
                silence_gap = next_start - end
                if silence_gap <= max_silence_samples:
                    processed_audio.append(y[end:next_start])

        y = np.concatenate(processed_audio)

    start_sample = int(start_time * target_sr)
    if max_duration is None:
        end_sample = len(y)
    else:
        end_sample = min(start_sample + int(max_duration * target_sr), len(y))

    y_window = y[start_sample:end_sample]
    total_samples = len(y_window)

    chunk_length = int(chunk_duration * target_sr)
    step_size = max(1, int((chunk_duration - overlap) * target_sr))

    chunks = []
    i = 0
    while i < total_samples:
        end_index = min(i + chunk_length, total_samples)
        chunk = y_window[i:end_index]

        actual_duration = len(chunk) / target_sr
        if actual_duration / chunk_duration >= min_chunk_ratio:
            if len(chunk) < chunk_length:
                pad_size = chunk_length - len(chunk)
                left_pad = pad_size // 2
                right_pad = pad_size - left_pad
                chunk = np.pad(chunk, (left_pad, right_pad), 'constant')
            chunks.append((chunk, label, patient_id, exercise))

        if end_index == total_samples:
            break

        i += step_size

    return chunks


def execute_preprocess_and_split(
    df,
    start_time=0,
    chunk_duration=5,
    max_duration=None,
    target_sr=16000,
    remove_silence=True,
    top_db=25,
    silence_duration=0.5,
    file_path_column='File_Path',
    label_column='Label',
    patient_column='Patient',
    audio_name_column='Audio_Name',
    overlap=0,
    min_chunk_ratio=0.7
):
    """
    Processes all audio files in a DataFrame and splits into chunks.

    Args:
        df (pd.DataFrame): DataFrame with audio metadata.
            Must contain 'Audio_Name' column for exercise type.
        start_time (float): Start time in seconds.
        chunk_duration (float): Duration of each chunk.
        max_duration (float, optional): Maximum duration to process.
        target_sr (int): Target sample rate.
        remove_silence (bool): Whether to remove silence.
        top_db (float): Silence threshold.
        silence_duration (float): Max silence duration to keep.
        file_path_column (str): Column name for file paths.
        label_column (str): Column name for labels.
        patient_column (str): Column name for patient IDs.
        audio_name_column (str): Column name for audio name (contains exercise type).
        overlap (float): Overlap between chunks.
        min_chunk_ratio (float): Minimum chunk length ratio.

    Returns:
        tuple: (chunks_np, labels_np, patient_ids_np, exercises_np)
            - chunks_np: Audio chunks
            - labels_np: Labels
            - patient_ids_np: Patient IDs
            - exercises_np: Exercise names (extracted from Audio_Name column)
    """
    all_chunks = []

    for _, row in df.iterrows():
        file_path = row[file_path_column]
        label = row[label_column]
        patient_id = row[patient_column]
        
        # Extract exercise from Audio_Name column
        # Audio_Name format: "001_PD" -> extract exercise part
        # If full format like "vocal_001_PD", extract just the exercise type
        audio_name = row.get(audio_name_column, 'unknown')
        
        # Try to extract exercise name (first part before underscore if format is "exercise_number_label")
        # Otherwise use the whole Audio_Name
        if audio_name and '_' in str(audio_name):
            parts = str(audio_name).split('_')
            if len(parts) >= 2 and parts[0] in ['vocal', 'text', 'reading', 'diado', 'patachaka']:
                exercise = parts[0]  # Known exercise types
            else:
                exercise = audio_name  # Use full name
        else:
            exercise = str(audio_name)

        chunks = process_and_split_audio(
            audio_path=file_path,
            label=label,
            patient_id=patient_id,
            exercise=exercise,
            start_time=start_time,
            chunk_duration=chunk_duration,
            max_duration=max_duration,
            target_sr=target_sr,
            remove_silence=remove_silence,
            top_db=top_db,
            silence_duration=silence_duration,
            overlap=overlap,
            min_chunk_ratio=min_chunk_ratio
        )
        all_chunks.extend(chunks)

    print(f"Total chunks generated: {len(all_chunks)}")

    if len(all_chunks) > 0:
        chunks, labels, ids, exercises = zip(*all_chunks)
        exercises_np = np.array(exercises)
        
        print(f"Unique exercises: {np.unique(exercises_np)}")
        
        return (
            np.array(chunks, dtype=np.float32),
            np.array(labels),
            np.array(ids),
            exercises_np
        )
    return np.array([]), np.array([]), np.array([]), np.array([])


def get_unique_exercises(df, audio_name_column='Audio_Name'):
    """
    Extract unique exercise types from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with audio metadata.
        audio_name_column (str): Column name for audio name.

    Returns:
        np.array: Unique exercise names.
    """
    exercises = []
    for audio_name in df[audio_name_column]:
        if audio_name and '_' in str(audio_name):
            parts = str(audio_name).split('_')
            if len(parts) >= 2 and parts[0] in ['vocal', 'text', 'reading', 'diado', 'patachaka']:
                exercises.append(parts[0])
            else:
                exercises.append(str(audio_name))
        else:
            exercises.append(str(audio_name))
    
    return np.unique(exercises)