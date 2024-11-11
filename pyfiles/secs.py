from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import wespeaker
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
import os
import numpy as np

### Objective Scores ###
import requests
from wespeaker.cli.hub import Hub, download

def download_wespeaker_model_local(lang: str, model_dir: str) -> str: # For Wespeaker downloading code
    #### WAVLM Download ####
    # cache_dir = "/mntcephfs/data/audiow/shoinoue/Model/hf_hub/wavlm/"
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv', cache_dir=cache_dir)
    # model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv', cache_dir=cache_dir)
    # HF_ENDPOINT=https://hf-mirror.com HF_HOME={cache_dir} python3 *.py
    
    if lang not in Hub.Assets.keys():
        print('ERROR: Unsupported lang {} !!!'.format(lang))
        sys.exit(1)
    model = Hub.Assets[lang]
    model_dir = os.path.join(model_dir, ".wespeaker", lang)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if set(["avg_model.pt",
            "config.yaml"]).issubset(set(os.listdir(model_dir))):
        return model_dir
    else:
        response = requests.get(
            "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"  # noqa
        )
        model_info = next(data for data in response.json()["Data"]
                          if data["Key"] == model)
        model_url = model_info['Url']
        download(model_url, model_dir)
        return model_dir

def cosine_similarity(e1, e2): # from wespeaker, delete the normalizing part
    cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
    cosine_score = cosine_score.item()
    # return (cosine_score + 1.0) / 2
    return cosine_score

class SpeechObjectiveEvaluation:
    def __init__(self, sr=16000, target_models=["speechmos", "wavlm", "wespeaker"],
                 # wavlm_path='../../UniSpeech/WavLM/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/',
                 wavlm_path='/mntcephfs/data/audiow/shoinoue/Model/hf_hub/wavlm/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/',
                 wespeaker_dir="/mntcephfs/data/audiow/shoinoue/Model/models/wespeaker",
                 device="cuda",
                ):
        self.target_models = target_models
        self.sr = sr
        self.device = "cuda"
        if "wavlm" in target_models:
            self.wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_path, local_files_only=True)
            self.wavlm = WavLMForXVector.from_pretrained(wavlm_path, local_files_only=True)
        self.wespeaker = {}
        if "wespeaker" in target_models:
            wespeaker_model_dir = download_wespeaker_model_local("english", wespeaker_dir)
            self.wespeaker["wespeaker"] = wespeaker.load_model_local(wespeaker_model_dir)
        if "wespeaker_lm" in target_models:
            self.wespeaker["wespeaker_lm"] = wespeaker.load_model_local(f'{wespeaker_dir}/voxceleb_resnet34_LM/')
        if "wespeaker_nolm" in target_models:
            self.wespeaker["wespeaker_nolm"] = wespeaker.load_model_local(f'{wespeaker_dir}/voxceleb_resnet34/')
        if "resemblyzer" in target_models:
            self.resemblyzer = VoiceEncoder()
        if "speechmos" in target_models:
            self.speechmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")
            
    def get_speaker_embedding(self, path):
        embs = {}
        if "wavlm" in self.target_models:
            self.wavlm = self.wavlm.to(self.device)
            
            audio = [librosa.load(p, sr=self.sr)[0] for p in [path]]
            inputs = self.wavlm_feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=self.sr).to(self.device)
            embeddings = self.wavlm(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
            cosine_sim = torch.nn.CosineSimilarity(dim=-1)
            embs["wavlm"] = embeddings[0].detach().cpu().numpy()
            
            self.wavlm = self.wavlm.to("cpu")
            
        for ws in ["wespeaker", "wespeaker_lm", "wespeaker_nolm"]:
            if ws in self.target_models:
                self.wespeaker[ws].model = self.wespeaker[ws].model.to(self.device)
                self.wespeaker[ws].device = self.device
                embs[ws] = self.wespeaker[ws].extract_embedding(path).detach().cpu().numpy()

                self.wespeaker[ws].model = self.wespeaker[ws].model.to("cpu")
                self.wespeaker[ws].device = "cpu"
            
        if "resemblyzer" in self.target_models:
            path = Path(path)
            wav = preprocess_wav(path, self.sr)
            embs["resemblyzer"] = self.resemblyzer.embed_utterance(wav)
            
        return embs
        
    def get_speaker_similarity(self, f1, f2):
        similarity = {}
        embs1 = self.get_speaker_embedding(f1)
        embs2 = self.get_speaker_embedding(f2)
        for mn in embs1:
            e1 = embs1[mn]
            e2 = embs2[mn]
            similarity[mn] = cosine_similarity(torch.tensor(e1), torch.tensor(e2))
        return similarity
    
    def get_speech_quality(self, path):
        quality = {}
        if "speechmos" in self.target_models:
            wave, _ = librosa.load(path, sr=self.sr, mono=True)
            quality["speechmos"] = np.array(self.speechmos(torch.from_numpy(wave).unsqueeze(0), self.sr).detach().cpu()).sum()
        return quality