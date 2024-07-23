import itertools
import os
import warnings
from typing import cast

import librosa
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import pyloudnorm
import sounddevice
import soundfile
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from audioseal.builder import create_generator
    from omegaconf import DictConfig
    from omegaconf import OmegaConf
    from speechbrain.pretrained import EncoderClassifier
    from torchaudio.transforms import Resample

from Architectures.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Architectures.Vocoder.HiFiGAN_Generator import HiFiGAN
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from Utility.storage_config import MODELS_DIR
from Utility.utils import cumsum_durations
from Utility.utils import float2pcm


class ToucanTTSInterface(torch.nn.Module):

    def __init__(
        self,
        device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
        tts_model_path=os.path.join(
            MODELS_DIR, f"ToucanTTS_Shan", "best.pt"
        ),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
        vocoder_model_path=os.path.join(
            MODELS_DIR, f"Vocoder", "best.pt"
        ),  # path to the Vocoder checkpoint
        language="eng",  # initial language of the model, can be changed later with the setter methods
        enhance=None,  # legacy argument
    ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(
                MODELS_DIR, f"ToucanTTS_{tts_model_path}", "best.pt"
            )
        if "USER" not in os.environ:
            os.environ["USER"] = (
                ""  # that's the case under Windows, but omegaconf needs this
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            watermark_conf = cast(
                DictConfig,
                OmegaConf.load("InferenceInterfaces/audioseal_wm_16bits.yaml"),
            )
            self.watermark = create_generator(watermark_conf)
            self.watermark.load_state_dict(
                torch.load("Models/audioseal/generator.pth", map_location="cpu")[
                    "model"
                ]
            )  # downloaded from https://dl.fbaipublicfiles.com/audioseal/6edcf62f/generator.pth originally

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(
            language=language, add_silence_to_end=True
        )

        #####################################
        #   load phone to features model    #
        #####################################
        checkpoint = torch.load(tts_model_path, map_location="cpu")
        self.phone2mel = ToucanTTS(
            weights=checkpoint["model"], config=checkpoint["config"]
        )
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))

        ######################################
        #  load features to style models     #
        ######################################
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
            savedir=os.path.join(
                MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"
            ),
        )

        ################################
        #  load mel to wave model      #
        ################################
        vocoder_checkpoint = torch.load(vocoder_model_path, map_location="cpu")
        self.vocoder = HiFiGAN()
        self.vocoder.load_state_dict(vocoder_checkpoint)
        self.vocoder = self.vocoder.to(device).eval()
        self.vocoder.remove_weight_norm()
        self.meter = pyloudnorm.Meter(24000)

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, device=device)
        self.phone2mel.eval()
        self.vocoder.eval()
        self.lang_id = get_language_id(language)
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        if type(path_to_reference_audio) != list:
            path_to_reference_audio = [path_to_reference_audio]

        if len(path_to_reference_audio) > 0:
            for path in path_to_reference_audio:
                assert os.path.exists(path)
            speaker_embs = list()
            for path in path_to_reference_audio:
                wave, sr = soundfile.read(path)
                if len(wave.shape) > 1:  # oh no, we found a stereo audio!
                    if (
                        len(wave[0]) == 2
                    ):  # let's figure out whether we need to switch the axes
                        wave = wave.transpose()  # if yes, we switch the axes.
                wave = librosa.to_mono(wave)
                wave = Resample(orig_freq=sr, new_freq=16000).to(self.device)(
                    torch.tensor(wave, device=self.device, dtype=torch.float32)
                )
                speaker_embedding = self.speaker_embedding_func_ecapa.encode_batch(
                    wavs=wave.to(self.device).squeeze().unsqueeze(0)
                ).squeeze()
                speaker_embs.append(speaker_embedding)
            self.default_utterance_embedding = sum(speaker_embs) / len(speaker_embs)

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(
            language=lang_id, add_silence_to_end=True
        )

    def set_accent_language(self, lang_id):
        if lang_id in [
            "ajp",
            "ajt",
            "lak",
            "lno",
            "nul",
            "pii",
            "plj",
            "slq",
            "smd",
            "snb",
            "tpw",
            "wya",
            "zua",
            "en-us",
            "en-sc",
            "fr-be",
            "fr-sw",
            "pt-br",
            "spa-lat",
            "vi-ctr",
            "vi-so",
        ]:
            if lang_id == "vi-so" or lang_id == "vi-ctr":
                lang_id = "vie"
            elif lang_id == "spa-lat":
                lang_id = "spa"
            elif lang_id == "pt-br":
                lang_id = "por"
            elif lang_id == "fr-sw" or lang_id == "fr-be":
                lang_id = "fra"
            elif lang_id == "en-sc" or lang_id == "en-us":
                lang_id = "eng"
            else:
                # no clue where these others are even coming from, they are not in ISO 639-2
                lang_id = "eng"

        self.lang_id = get_language_id(lang_id).to(self.device)

    def forward(
        self,
        text,
        view=False,
        duration_scaling_factor=1.0,
        pitch_variance_scale=1.0,
        energy_variance_scale=1.0,
        pause_duration_scaling_factor=1.0,
        durations=None,
        pitch=None,
        energy=None,
        input_is_phones=False,
        return_plot_as_filepath=False,
        loudness_in_db=-24.0,
        glow_sampling_temperature=0.2,
    ):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(
                text, input_phonemes=input_is_phones
            ).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(
                phones,
                return_duration_pitch_energy=True,
                utterance_embedding=self.default_utterance_embedding,
                durations=durations,
                pitch=pitch,
                energy=energy,
                lang_id=self.lang_id,
                duration_scaling_factor=duration_scaling_factor,
                pitch_variance_scale=pitch_variance_scale,
                energy_variance_scale=energy_variance_scale,
                pause_duration_scaling_factor=pause_duration_scaling_factor,
                glow_sampling_temperature=glow_sampling_temperature,
            )

            wave, _, _ = self.vocoder(mel.unsqueeze(0))
            wave = wave.squeeze().cpu()
        wave = wave.numpy()
        sr = 24000
        try:
            loudness = self.meter.integrated_loudness(wave)
            wave = pyloudnorm.normalize.loudness(wave, loudness, loudness_in_db)
        except ValueError:
            # if the audio is too short, a value error will arise
            pass
        with torch.inference_mode():
            wave = (
                (
                    torch.tensor(wave)
                    + 0.1
                    * self.watermark.get_watermark(
                        torch.tensor(wave).to(self.device).unsqueeze(0).unsqueeze(0)
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                )
                .detach()
                .numpy()
            )

        if view or return_plot_as_filepath:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

            # fpath = "./src/fonts/Shan.ttf"
            fpath = os.path.join(os.path.dirname(__file__), "src/fonts/Shan.ttf")
            prop = fm.FontProperties(fname=fpath)

            ax.imshow(mel.cpu().numpy(), origin="lower", cmap="GnBu")
            ax.yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax.xaxis.grid(True, which="minor")
            ax.set_xticks(label_positions, minor=False)
            if input_is_phones:
                phones = text.replace(" ", "|")
            else:
                phones = self.text2phone.get_phone_string(text, for_plot_labels=True)
            ax.set_xticklabels(phones)
            word_boundaries = list()
            for label_index, phone in enumerate(phones):
                if phone == "|":
                    word_boundaries.append(label_positions[label_index])

            try:
                prev_word_boundary = 0
                word_label_positions = list()
                for word_boundary in word_boundaries:
                    word_label_positions.append(
                        (word_boundary + prev_word_boundary) / 2
                    )
                    prev_word_boundary = word_boundary
                word_label_positions.append(
                    (duration_splits[-1] + prev_word_boundary) / 2
                )

                secondary_ax = ax.secondary_xaxis("bottom")
                secondary_ax.tick_params(axis="x", direction="out", pad=24)
                secondary_ax.set_xticks(word_label_positions, minor=False)
                secondary_ax.set_xticklabels(text.split(), fontproperties=prop)
                secondary_ax.tick_params(axis="x", colors="orange")
                secondary_ax.xaxis.label.set_color("orange")
            except ValueError:
                ax.set_title(text)
            except IndexError:
                ax.set_title(text)

            ax.vlines(
                x=duration_splits,
                colors="green",
                linestyles="solid",
                ymin=0,
                ymax=120,
                linewidth=0.5,
            )
            ax.vlines(
                x=word_boundaries,
                colors="orange",
                linestyles="solid",
                ymin=0,
                ymax=120,
                linewidth=1.0,
            )
            plt.subplots_adjust(
                left=0.02, bottom=0.2, right=0.98, top=0.9, wspace=0.0, hspace=0.0
            )
            ax.set_aspect("auto")

            if return_plot_as_filepath:
                plt.savefig("tmp.png")
                return wave, sr, "tmp.png"
        return wave, sr

    def read_to_file(
        self,
        text_list,
        file_location,
        duration_scaling_factor=1.0,
        pitch_variance_scale=1.0,
        energy_variance_scale=1.0,
        pause_duration_scaling_factor=1.0,
        silent=False,
        dur_list=None,
        pitch_list=None,
        energy_list=None,
        glow_sampling_temperature=0.2,
    ):
        """
        Args:
            silent: Whether to be verbose about the process
            text_list: A list of strings to be read
            file_location: The path and name of the file it should be saved to
            energy_list: list of energy tensors to be used for the texts
            pitch_list: list of pitch tensors to be used for the texts
            dur_list: list of duration tensors to be used for the texts
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        if not dur_list:
            dur_list = []
        if not pitch_list:
            pitch_list = []
        if not energy_list:
            energy_list = []
        silence = torch.zeros([14300])
        wav = silence.clone()
        for text, durations, pitch, energy in itertools.zip_longest(
            text_list, dur_list, pitch_list, energy_list
        ):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                spoken_sentence, sr = self(
                    text,
                    durations=(
                        durations.to(self.device) if durations is not None else None
                    ),
                    pitch=pitch.to(self.device) if pitch is not None else None,
                    energy=energy.to(self.device) if energy is not None else None,
                    duration_scaling_factor=duration_scaling_factor,
                    pitch_variance_scale=pitch_variance_scale,
                    energy_variance_scale=energy_variance_scale,
                    pause_duration_scaling_factor=pause_duration_scaling_factor,
                    glow_sampling_temperature=glow_sampling_temperature,
                )
                spoken_sentence = torch.tensor(spoken_sentence).cpu()
                wav = torch.cat((wav, spoken_sentence, silence), 0)
        soundfile.write(
            file=file_location, data=float2pcm(wav), samplerate=sr, subtype="PCM_16"
        )

    def read_aloud(
        self,
        text,
        view=False,
        duration_scaling_factor=1.0,
        pitch_variance_scale=1.0,
        energy_variance_scale=1.0,
        blocking=False,
        glow_sampling_temperature=0.2,
    ):
        if text.strip() == "":
            return
        wav, sr = self(
            text,
            view,
            duration_scaling_factor=duration_scaling_factor,
            pitch_variance_scale=pitch_variance_scale,
            energy_variance_scale=energy_variance_scale,
            glow_sampling_temperature=glow_sampling_temperature,
        )
        silence = torch.zeros([sr // 2])
        wav = torch.cat((silence, torch.tensor(wav), silence), 0).numpy()
        sounddevice.play(float2pcm(wav), samplerate=sr)
        if view:
            plt.show()
        if blocking:
            sounddevice.wait()
