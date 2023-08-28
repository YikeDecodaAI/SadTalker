import os
import tempfile
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import nltk
import numpy as np
import struct  # Import the struct module for PCM conversion
from transformers import BarkModel, AutoProcessor
import torch
from optimum.bettertransformer import BetterTransformer
from elevenlabs import voices, set_api_key, generate, save
from datetime import datetime


class TTSTalker():
    def __init__(self) -> None:
        self.silence = np.zeros(int(0.25 * SAMPLE_RATE))
        self.speaker_actions = {
            "老男人": "v2/en_speaker_6",
            "小女孩": "v2/en_speaker_9",
            "中年男人": "v2/en_speaker_1",
            "老太太": "en_speaker_4"
        }

    def test(self, script, speaker):
        sentences = nltk.sent_tokenize(script)
        processor = AutoProcessor.from_pretrained("suno/bark-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
        model = BetterTransformer.transform(model, keep_original_model=False)
        model.enable_cpu_offload()

        pieces = []
        for sentence in sentences:
            inputs = processor(sentence, voice_preset=self.speaker_actions[speaker]).to(device)
            audio_array = model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            pieces += [audio_array, self.silence.copy()]

        # 将所有音频片段连接起来
        final_audio = np.concatenate(pieces)

        # 创建一个临时文件来保存.wav音频
        tempf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

        # Convert float audio to PCM format
        pcm_data = (final_audio * 32767).astype(np.int16)

        write_wav(tempf.name, SAMPLE_RATE, pcm_data)

        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()

        return tempf.name


class ElevenLabsTTS():
    def __init__(self):
        set_api_key("76d9ba7561aed7a7d8ebcc5784c0d5bb")
        self.voice_id = {
            "Charli": 39,
            "Obama": 40,
            "Biden": 41,
            "Trump": 42
        }

    def get_voice(self, voice_id):
        vs = voices()
        return vs[voice_id]

    def generate_audio(self, script, speaker):
        voice = self.get_voice(self.voice_id[speaker])
        # Get current time and format it as a string
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create the .mp3 filename with the current time
        outfile = os.path.join(current_time + '.mp3')
        audio = generate(text=script, voice=voice)
        save(audio, outfile)

        return outfile
