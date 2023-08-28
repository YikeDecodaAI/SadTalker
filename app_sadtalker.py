import os, sys
import gradio as gr
from src.gradio_demo import SadTalker
from src.utils.text2speech import TTSTalker, ElevenLabsTTS
tts_talker = TTSTalker()
eleven_labs_tts = ElevenLabsTTS()

try:
    import webui  # in webui

    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)


def generate_audio_callback(input_text, speaker_selected):
    # 检查输入文本
    if not input_text:
        return False
    # 根据选择的音色决定使用哪个TTS引擎
    if speaker_selected in ["老男人", "小女孩", "中年男人", "老太太"]:
        return tts_talker.test(input_text, speaker_selected)
    elif speaker_selected in ["Charli", "Obama", "Biden", "Trump"]:
        return eleven_labs_tts.generate_audio(input_text, speaker_selected)


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> Decoda.AI 给定音频让图片说话 </span> </h2>")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('上传人像图片'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath",
                                                    elem_id="img2img_image").style(width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('上传音频或者在下面的文本框输入文字去生成音频'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="输入文字去生成音频", lines=5,
                                                        placeholder="请输入英文，我们会将输入转化为TTS，暂时不支持中文，目前生成1分半以内的视频没有问题，需要保证由文本生成的音频或者自己上传的音频不要超过1分半")
                                tts = gr.Button('Generate audio', elem_id="sadtalker_audio_generate", variant='primary')
                                speaker_selected = gr.Radio(
                                    ["老男人", "小女孩", "中年男人", "老太太", "Charli", "Obama", "Biden", "Trump"],
                                    label="请选择音色")
                                tts.click(fn=generate_audio_callback, inputs=[input_text, speaker_selected],
                                          outputs=[driven_audio])

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('设置'):
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0,
                                                   info="脸部表情，推荐选择0或1两种表情，看起来自然")  #
                            size_of_image = gr.Radio([256, 512], value=512, label='face model resolution',
                                                     info="默认选512，效果好，但是生成视频所需的时间长")  #
                            preprocess_type = gr.Radio(['crop', 'full'], value='full', label='preprocess',
                                                       info="crop只生成头像，full生成完整图片，默认选full")
                            is_still_mode = gr.Checkbox(
                                label="Still Mode (fewer hand motion, works with preprocess `full`)", value=True,
                                info="上面选项选full，这个就要勾选，选crop，这个不用勾选")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2,
                                                   info="代表生成视频的速度，数值越高，生成越快，一般推荐选2，稳定")
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer",
                                                   info="生成的视频脸不清楚的时候需要勾选")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

        if warpfn:
            submit.click(
                fn=warpfn(sad_talker.test),
                inputs=[source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,
                        size_of_image,
                        pose_style
                        ],
                outputs=[gen_video]
            )
        else:
            submit.click(
                fn=sad_talker.test,
                inputs=[source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,
                        size_of_image,
                        pose_style
                        ],
                outputs=[gen_video]
            )

    return sadtalker_interface


if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=5000, share=True)
