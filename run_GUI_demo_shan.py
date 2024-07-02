import os
import gradio as gr
import torch
import torch.cuda
from Utility.utils import float2pcm
from Architectures.ControllabilityGAN.GAN import GanWrapper
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import MODELS_DIR
from Utility.utils import load_json_from_path


demo = gr.Blocks()


class ControllableInterface:

    def __init__(self, gpu_id="cpu", available_artificial_voices=1000):
        if gpu_id == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ToucanTTSInterface(device=self.device, tts_model_path="Shan")
        self.wgan = GanWrapper(
            os.path.join(MODELS_DIR, "Embedding", "embedding_gan.pt"),
            device=self.device,
        )
        self.generated_speaker_embeds = list()
        self.available_artificial_voices = available_artificial_voices
        self.current_language = ""
        self.current_accent = ""

    def read(
        self,
        prompt,
        language,
        accent,
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
        emb_slider_6,
    ):
        if self.current_language != language:
            self.model.set_phonemizer_language(language)
            self.current_language = language
        if self.current_accent != accent:
            self.model.set_accent_language(accent)
            self.current_accent = accent

        self.wgan.set_latent(voice_seed)
        controllability_vector = torch.tensor(
            [
                emb_slider_1,
                emb_slider_2,
                emb_slider_3,
                emb_slider_4,
                emb_slider_5,
                emb_slider_6,
            ],
            dtype=torch.float32,
        )
        embedding = self.wgan.modify_embed(controllability_vector)
        self.model.set_utterance_embedding(embedding=embedding)

        phones = self.model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            if language == "deu":
                prompt = "Deine Eingabe war zu lang. Bitte versuche es entweder mit einem kürzeren Text oder teile ihn in mehrere Teile auf."
            elif language == "ell":
                prompt = "Η εισήγησή σας ήταν πολύ μεγάλη. Παρακαλώ δοκιμάστε είτε ένα μικρότερο κείμενο είτε χωρίστε το σε διάφορα μέρη."
            elif language == "spa":
                prompt = "Su entrada es demasiado larga. Por favor, intente un texto más corto o divídalo en varias partes."
            elif language == "fin":
                prompt = "Vastauksesi oli liian pitkä. Kokeile joko lyhyempää tekstiä tai jaa se useampaan osaan."
            elif language == "rus":
                prompt = "Ваш текст слишком длинный. Пожалуйста, попробуйте либо сократить текст, либо разделить его на несколько частей."
            elif language == "hun":
                prompt = "Túl hosszú volt a bevitele. Kérjük, próbáljon meg rövidebb szöveget írni, vagy ossza több részre."
            elif language == "nld":
                prompt = "Uw input was te lang. Probeer een kortere tekst of splits het in verschillende delen."
            elif language == "fra":
                prompt = "Votre saisie était trop longue. Veuillez essayer un texte plus court ou le diviser en plusieurs parties."
            elif language == "pol":
                prompt = "Twój wpis był zbyt długi. Spróbuj skrócić tekst lub podzielić go na kilka części."
            elif language == "por":
                prompt = "O seu contributo foi demasiado longo. Por favor, tente um texto mais curto ou divida-o em várias partes."
            elif language == "ita":
                prompt = "Il tuo input era troppo lungo. Per favore, prova un testo più corto o dividilo in più parti."
            elif language == "cmn":
                prompt = "你的输入太长了。请尝试使用较短的文本或将其拆分为多个部分。"
            elif language == "vie":
                prompt = "Đầu vào của bạn quá dài. Vui lòng thử một văn bản ngắn hơn hoặc chia nó thành nhiều phần."
            else:
                prompt = "Your input was too long. Please try either a shorter text or split it into several parts."
                if self.current_language != "eng":
                    self.model.set_phonemizer_language("eng")
                    self.current_language = "eng"
                if self.current_accent != "eng":
                    self.model.set_accent_language("eng")
                    self.current_accent = "eng"

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

    def __init__(
        self,
        gpu_id="cpu",
        title="Controllable Text-to-Speech for over 7000 Languages",
        article="",
        available_artificial_voices=1000,
        path_to_iso_list="Preprocessing/multilinguality/iso_to_fullname.json",
    ):
        iso_to_name = load_json_from_path(path_to_iso_list)
        text_selection = [
            f"{iso_to_name[iso_code]} Text ({iso_code})" for iso_code in iso_to_name
        ]
        # accent_selection = [f"{iso_to_name[iso_code]} Accent ({iso_code})" for iso_code in iso_to_name]

        self.controllable_ui = ControllableInterface(
            gpu_id=gpu_id, available_artificial_voices=available_artificial_voices
        )
        self.iface = gr.Interface(
            fn=self.read,
            inputs=[
                gr.Textbox(
                    lines=2,
                    placeholder="write what you want the synthesis to read here...",
                    value="The woods are lovely, dark and deep, but I have promises to keep, and miles to go, before I sleep.",
                    label="Text input",
                ),
                gr.Dropdown(
                    text_selection,
                    type="value",
                    value="English Text (eng)",
                    label="Select the Language of the Text (type on your keyboard to find it quickly)",
                ),
                gr.Slider(
                    minimum=0,
                    maximum=available_artificial_voices,
                    step=1,
                    value=279,
                    label="Random Seed for the artificial Voice",
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

    def read(
        self,
        prompt,
        language,
        voice_seed,
        duration_scaling_factor,
        pitch_variance_scale,
        energy_variance_scale,
        emb1,
        emb2,
    ):
        sr, wav, fig = self.controllable_ui.read(
            prompt=prompt,
            language=language.split(" ")[-1].split("(")[1].split(")")[0],
            accent=language.split(" ")[-1].split("(")[1].split(")")[0],
            voice_seed=voice_seed,
            duration_scaling_factor=duration_scaling_factor,
            pause_duration_scaling_factor=1.0,
            pitch_variance_scale=pitch_variance_scale,
            energy_variance_scale=energy_variance_scale,
            emb_slider_1=emb1,
            emb_slider_2=emb2,
            emb_slider_3=0.0,
            emb_slider_4=0.0,
            emb_slider_5=0.0,
            emb_slider_6=0.0,
        )
        return (sr, float2pcm(wav)), fig

    def render(self):
        return self.iface


if __name__ == "__main__":
    with gr.Blocks() as demo:
        TTSWebUI(gpu_id="cuda" if torch.cuda.is_available() else "cpu").render()

    demo.launch()
