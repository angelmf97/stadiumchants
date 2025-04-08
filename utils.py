import pandas as pd
import music21
from music21 import stream, note, instrument
import numpy as np

def csv_to_chorale_score(csv_path, quarter_length=1.0, voice_names=None):
    """
    Convert a CSV file with MIDI pitch values into a music21 Score.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - quarter_length (float): Duration of each note in quarterLength units.
    - voice_names (list of str): Optional custom names for the four voices. Defaults to SATB.

    Returns:
    - music21.stream.Score: A score with four parts representing the chorale.
    """
    if voice_names is None:
        voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

    df = pd.read_csv(csv_path)
    score = stream.Score()

    for col_idx, col in enumerate(df.columns):
        part = stream.Part()
        part.id = voice_names[col_idx]
        part.insert(0, instrument.fromString(voice_names[col_idx]))  # Optional instrument assignment

        for midi_pitch in df[col]:
            if pd.isna(midi_pitch):
                n = note.Rest(quarterLength=quarter_length)
            else:
                n = note.Note(int(midi_pitch), quarterLength=quarter_length)
            part.append(n)
        
        score.append(part)
        break
    score.makeMeasures(inPlace=True)
    return score.parts[0]

def analyze_intervals(stream):

    # Get the notes in the stream
    notes = stream.flatten().notes
    
    # Calculate the intervals between consecutive notes
    intervals = []
    for i in range(1, len(notes)):
        interval = music21.interval.Interval(notes[i-1], notes[i])
        intervals.append(interval.semitones)

    # Calculate the mean interval distance
    mean_interval = np.mean(np.abs(intervals))
    
    return mean_interval

def analyze_unique_notes_per_bar(stream):
    # Get the notes in the stream
    notes = stream.flatten().notes
    
    # Group notes by measure
    measures = stream.getElementsByClass('Measure')
    
    unique_notes_per_bar = []
    
    for measure in measures:
        unique_notes = set()
        for note in measure.notes:
            unique_notes.add(note.nameWithOctave)
            
        unique_notes_per_bar.append(len(unique_notes))
    
    return np.mean(unique_notes_per_bar)

def analyze_accidentals(stream):
    # Get the notes in the stream
    notes = stream.flatten().notes
    
    measures = stream.getElementsByClass('Measure')
    
    accidentals = []
    
    for measure in measures:
        n_accidentals = 0
        for note in measure.notes:
            if note.pitch.accidental is not None:
                n_accidentals += 1
    
        accidentals.append(n_accidentals)
    return len(accidentals)

def analyze_syncopations(stream):

    # Get the notes in the stream
    notes = stream.flatten().notes

    measures = stream.getElementsByClass('Measure')
    
    # Calculate the syncopation based on the note durations
    syncopations = []

    for measure in measures:
        n_sync = 0
        for i in range(1, len(measure.notes)):
            if measure.notes[i].quarterLength < 1.0 and measure.notes[i-1].quarterLength >= 1.0:
                n_sync += 1
        syncopations.append(n_sync)
    
    return np.mean(syncopations)

def analyze_unique_durations(stream):
    # Get the notes in the stream
    notes = stream.flatten().notes
    
    measures = stream.getElementsByClass('Measure')

    unique_durations = list()
    for measure in measures:
        # Calculate the unique durations
        unique_durations_measure = set()
        for note in measure.notes:
            unique_durations_measure.add(note.quarterLength)
        unique_durations.append(len(unique_durations_measure))
    
    return np.mean(unique_durations)

def analyze_notes_per_bar(stream):
    # Get the notes in the stream
    notes = stream.flatten().notes
    
    measures = stream.getElementsByClass('Measure')
    
    notes_per_bar = []
    
    for measure in measures:
        n_notes = 0
        for note in measure.notes:
            n_notes += 1
        notes_per_bar.append(n_notes)
    
    return np.mean(notes_per_bar)

def get_time_signature(stream):
    # Get the time signature of the stream
    time_signature = stream.getElementsByClass('TimeSignature')
    
    if len(time_signature) > 0:
        return time_signature[0].numerator, time_signature[0].denominator
    else:
        return None

def analyze_stream(stream):

    stream.makeMeasures(inPlace=True)

    results = dict()

    try:
    
        # Calculate the mean interval distance
        results["mean_interval"] = analyze_intervals(stream)
        
        # Calculate the number of unique notes per bar
        results["unique_notes_per_bar"] = analyze_unique_notes_per_bar(stream)
        
        # Calculate the number of accidentals
        results["accidentals"] = analyze_accidentals(stream)

        # Calculate the number of syncopations
        results["syncopations"] = analyze_syncopations(stream)

        # Calculate the number of unique durations
        results["unique_durations"] = analyze_unique_durations(stream)

        # Calculate the number of notes per bar
        results["notes_per_bar"] = analyze_notes_per_bar(stream)

        # Get the time signature
        time_signature = get_time_signature(stream)
        if time_signature:
            results["time_signature"] = f"{time_signature[0]}/{time_signature[1]}"
        else:
            results["time_signature"] = None
    
    except AttributeError:
        return None

    return results

# Function to add statistical significance annotations
def add_stat_significance(ax, pairs, p_values, y_offset=1, y_increment=0.5, h=0.2, show_non_significant=False):
    """
    Adds statistical significance annotations to the plot.
    """
    annot = ["*", "**", "***"]
    value = [0.05, 0.01, 0.001]
    ymax = ax.get_ylim()[1]
    for (pair, p_value) in zip(pairs, p_values):
        x1, x2 = pair
        y = ymax + y_offset
        if p_value < 0.05:
            annot_index = np.searchsorted(value, p_value)
            annot_text = f"{annot[annot_index]} (p<{value[annot_index]})"
            ax.text((x1 + x2) * 0.5, y + h, annot_text, ha='center', va='bottom', color='k')
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')

        elif show_non_significant:
            ax.text((x1 + x2) * 0.5, y + h, "ns", ha='center', va='bottom', color='k', bbox=dict(facecolor='w', edgecolor='w', pad=-1.5))
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
        y_offset += y_increment
    return ax   

