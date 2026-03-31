# src/custom_vad.py
import numpy as np
import librosa
import json
import os
from datetime import datetime

class CustomVoiceActivityDetector:
    """
    Custom Voice Activity Detection using Energy and Zero-Crossing Rate
    This is our own implementation to compare with pyannote
    """
    
    def __init__(self, frame_length=0.025, frame_shift=0.010, 
                 energy_threshold=0.01, zcr_threshold=(0.02, 0.5)):
        self.frame_length = frame_length  # 25ms frames
        self.frame_shift = frame_shift    # 10ms shift
        self.energy_threshold = energy_threshold
        self.zcr_low, self.zcr_high = zcr_threshold
        
    def detect_speech(self, audio_path):
        """
        Detect speech segments using energy and ZCR
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate frame indices
        frame_len = int(self.frame_length * sr)
        frame_shift = int(self.frame_shift * sr)
        
        num_frames = (len(y) - frame_len) // frame_shift + 1
        
        speech_frames = []
        frame_energies = []
        frame_zcrs = []
        
        for i in range(num_frames):
            start = i * frame_shift
            end = start + frame_len
            frame = y[start:end]
            
            # Calculate frame energy (log scale for better sensitivity)
            energy = np.sum(frame**2) / len(frame)
            log_energy = np.log10(energy + 1e-10)
            frame_energies.append(log_energy)
            
            # Calculate zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            frame_zcrs.append(zcr)
            
            # Simple rule: high energy + appropriate ZCR = speech
            is_speech = (log_energy > self.energy_threshold) and \
                        (self.zcr_low < zcr < self.zcr_high)
            speech_frames.append(is_speech)
        
        # Convert frame-level decisions to segments
        segments = self._frames_to_segments(speech_frames, frame_shift)
        
        # Post-processing
        segments = self._merge_close_segments(segments, gap=0.3)
        segments = self._filter_short_segments(segments, min_duration=0.5)
        
        return segments, {
            'total_frames': num_frames,
            'speech_frames': sum(speech_frames),
            'avg_energy': np.mean(frame_energies),
            'avg_zcr': np.mean(frame_zcrs)
        }
    
    def _frames_to_segments(self, speech_frames, frame_shift):
        """
        Convert frame-level decisions to time segments
        """
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                end_frame = i
                start_time = start_frame * frame_shift
                end_time = end_frame * frame_shift
                segments.append({
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'duration': round(end_time - start_time, 2)
                })
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            end_time = len(speech_frames) * frame_shift
            start_time = start_frame * frame_shift
            segments.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'duration': round(end_time - start_time, 2)
            })
        
        return segments
    
    def _merge_close_segments(self, segments, gap=0.3):
        """
        Merge segments that are very close together
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for seg in segments[1:]:
            if seg['start'] - current['end'] < gap:
                # Merge segments
                current['end'] = seg['end']
                current['duration'] = round(current['end'] - current['start'], 2)
            else:
                merged.append(current)
                current = seg
        
        merged.append(current)
        return merged
    
    def _filter_short_segments(self, segments, min_duration=0.5):
        """
        Remove segments shorter than min_duration
        """
        return [seg for seg in segments if seg['duration'] >= min_duration]
    
    def calculate_metrics(self, custom_segments, pyannote_segments):
        """
        Calculate comparison metrics between custom VAD and pyannote
        """
        # Convert to sets for comparison
        custom_times = set()
        pyannote_times = set()
        
        # Create time masks (rounded to 0.1s intervals)
        for seg in custom_segments:
            for t in np.arange(seg['start'], seg['end'], 0.1):
                custom_times.add(round(t, 1))
        
        for seg in pyannote_segments:
            for t in np.arange(seg['start'], seg['end'], 0.1):
                pyannote_times.add(round(t, 1))
        
        # Calculate metrics
        intersection = len(custom_times & pyannote_times)
        union = len(custom_times | pyannote_times)
        
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        
        # Precision and Recall (treating pyannote as ground truth)
        precision = intersection / len(custom_times) if len(custom_times) > 0 else 0
        recall = intersection / len(pyannote_times) if len(pyannote_times) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Segment count difference
        segment_diff = len(custom_segments) - len(pyannote_segments)
        
        return {
            'iou': round(iou, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1_score, 3),
            'custom_segments': len(custom_segments),
            'pyannote_segments': len(pyannote_segments),
            'segment_diff': segment_diff,
            'agreement_percentage': round(iou * 100, 1)
        }