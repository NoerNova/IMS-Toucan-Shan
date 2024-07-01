import os
import gradio as gr
import torch
import torch.cuda
from Utility.utils import float2pcm
from Architectures.ControllabilityGAN.GAN import GanWrapper
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import MODELS_DIR


class ControllableInterface:

    def __init__(self, gpu_id="cpu"):
        if gpu_id == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ToucanTTSInterface(
            device=self.device, tts_model_path="Finetuning_Shan"
        )
        self.wgan = GanWrapper(
            os.path.join(MODELS_DIR, "Embedding", "embedding_gan.pt"),
            device=self.device,
        )
        self.generated_speaker_embeds = list()

    def read(
        self,
        prompt,
        voice_seed,
        duration_scaling_factor,
        pause_duration_scaling_factor,
        pitch_variance_scale,
        energy_variance_scale,
        emb_slider_1,
        emb_slider_2,
        emb_slider_3,
        emb_slider_4,
        emb_slider_5,
    ):

        self.wgan.set_latent(voice_seed)
        controllability_vector = torch.tensor(
            [emb_slider_1, emb_slider_2, emb_slider_3, emb_slider_4, emb_slider_5],
            dtype=torch.float32,
        )
        embedding = self.wgan.modify_embed(controllability_vector)
        self.model.set_utterance_embedding(embedding=embedding)

        phones = self.model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            prompt = "Your input was too long. Please try either a shorter text or split it into several parts."

        print(prompt)
        wav, sr, fig = self.model(
            prompt,
            input_is_phones=False,
            duration_scaling_factor=duration_scaling_factor,
            pitch_variance_scale=pitch_variance_scale,
            energy_variance_scale=energy_variance_scale,
            pause_duration_scaling_factor=pause_duration_scaling_factor,
            return_plot_as_filepath=True,
        )
        return sr, wav, fig


class TTSWebUI:

    def __init__(self, gpu_id="cpu", title="ToucanTTS Shan Demo", article=""):
        self.controllable_ui = ControllableInterface(gpu_id=gpu_id)
        self.iface = gr.Interface(
            fn=self.read,
            inputs=[
                gr.Textbox(
                    lines=2,
                    placeholder="write what you want the synthesis to read here...",
                    value="",
                    label="Text input",
                ),
                gr.Slider(
                    minimum=0.7,
                    maximum=1.3,
                    step=0.1,
                    value=1.0,
                    label="Duration Scale",
                ),
                gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    step=0.1,
                    value=1.0,
                    label="Pitch Variance Scale",
                ),
                gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    step=0.1,
                    value=1.0,
                    label="Energy Variance Scale",
                ),
                gr.Slider(
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Femininity / Masculinity",
                ),
                gr.Slider(
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Voice Depth",
                ),
            ],
            outputs=[
                gr.Audio(type="numpy", label="Speech"),
                gr.Image(label="Visualization"),
            ],
            title=title,
            theme="default",
            allow_flagging="never",
            article=article,
        )
        self.iface.launch()

    def read(
        self,
        prompt,
        voice_seed,
        duration_scaling_factor,
        pitch_variance_scale,
        energy_variance_scale,
    ):
        sr, wav, fig = self.controllable_ui.read(
            prompt=prompt,
            voice_seed=voice_seed,
            duration_scaling_factor=duration_scaling_factor,
            pause_duration_scaling_factor=1.0,
            pitch_variance_scale=pitch_variance_scale,
            energy_variance_scale=energy_variance_scale,
            emb_slider_1=0.0,
            emb_slider_2=0.0,
            emb_slider_3=0.0,
            emb_slider_4=0.0,
            emb_slider_5=0.0,
        )
        return (sr, float2pcm(wav)), fig


if __name__ == "__main__":
    TTSWebUI(gpu_id="cuda" if torch.cuda.is_available() else "cpu")
