import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
txt_content = (
    "Incident : Manifestation et travaux entraînant une déviation de la ligne, "
    "Arrêts entre Nogent-le-Perreux(RER) et Place du Général Leclerc, "
    "et entre Château de Villemomble et Plateau d'Avron. "
    "Place de Stalingrad non desservie en direction de Château de Vincennes"
)

#wav = tts.tts(text = txt_content
#, speaker_wav="my/cloning/les-lectures-de-simone-un-nouveau-pneu.wav", language="fr",speed=0.9)
# Text to speech to a file
tts.tts_to_file(text = txt_content
, speaker_wav="my/cloning/les-lectures-de-simone-un-nouveau-pneu.wav", language="fr", file_path="output.wav",speed=0.9)

"""
from TTS.api import TTS

# Load the pre-trained French model (VITS)
tts = TTS("tts_models/fr/css10/vits")

# Synthesize speech from text in French
tts.tts_to_file(text=txt_content,speaker_wav="my/cloning/les-lectures-de-simone-un-nouveau-pneu.wav")
"""