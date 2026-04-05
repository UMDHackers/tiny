import torchaudio
from encodec import EncodecModel

class AudioEngine:
    def __init__(self):
        # High-fidelity 48kHz model
        self.model = EncodecModel.encodec_model_48khz()
    
    def encode(self, audio_path):
        # Load audio -> Encode to codes -> Zstd pack
        wav, sr = torchaudio.load(audio_path)
        # Model handles the heavy lifting
        with torch.no_grad():
            frames = self.model.encode(wav.unsqueeze(0))
        return frames

    def decode(self, codes):
        # Codes -> Waveform -> Save as .wav or .flac
        return self.model.decode(codes)
