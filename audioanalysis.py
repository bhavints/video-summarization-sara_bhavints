from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
audio_path = "../project_files/project_dataset/audio/soccer.wav"
[Fs, x] = aIO.read_audio_file(audio_path)
segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window = 1.0, weight = 0.3, plot = True)