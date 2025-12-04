import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device: str = "cpu"
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# model = ChatterboxTTS.from_pretrained(device=device)

# text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
# wav = model.generate(text)
# ta.save("test-1.wav", wav, model.sr)

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
text = "Idag skiner solen över Jönköping och temperaturen ligger på tjugotvå grader"
# wav = multilingual_model.generate(text, language_id="sv", audio_prompt_path="outputfile.mp3")
wav = multilingual_model.generate(text, language_id="sv")
ta.save("test-2.wav", wav, multilingual_model.sr)


# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-3.wav", wav, model.sr)
