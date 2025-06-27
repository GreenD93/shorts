import streamlit as st
from PIL import Image
import io
import tempfile
import os
import time
from threading import Thread

from src.shorts_composer import ShortsComposer

# API 키 설정
OPENAI_KEY = ""
LUMA_KEY = ""

AWS_ACCESS_KEY = ""
AWS_SECRET_KEY =  ""

st.set_page_config(layout="wide")
st.title("🎬 Shorts 자동 생성기")
st.markdown("""
이 앱은 입력한 프롬프트를 기반으로 자동으로 Shorts 영상을 생성합니다.\
이미지를 업로드하면 이미지 기반 스타일도 반영됩니다 (선택).
""")

with st.container():
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        image_prompt = st.text_area("🖼️ 이미지 생성 프롬프트", placeholder="예: 노을지는 해변에서 춤추는 사람")
        custom_instruction = st.text_area("📝 자막 커스텀 요청사항", placeholder="예: 케이뱅크에서 새로나온 ONE체크카드를 소개하는 쇼츠 영상 자막 8줄 만들어줘")

        subtitle_guideline = """TTS로 말했을 때 총 발화 시간이 8.5초 이내여야 해.
각 줄은 되도록 8자 이내, 짧고 리듬감 있게 만들어줘.
말했을 때 자연스럽고 부드러운 흐름이 되도록 해줘.
번호나 따옴표 없이, 줄바꿈으로만 구분해줘.

"""
        subtitle_prompt = subtitle_guideline + custom_instruction if custom_instruction else ""

        uploaded_image = st.file_uploader("📎 참고용 이미지 (선택)", type=["png", "jpg", "jpeg"])
        generate_clicked = st.button("🎥 Shorts 생성하기")

    with col2:
        if 'result_video_bytes' not in st.session_state:
            st.session_state.result_video_bytes = None

        if generate_clicked:
            if not image_prompt or not subtitle_prompt:
                st.error("이미지 프롬프트와 자막 프롬프트를 모두 입력해주세요.")
            else:
                progress_text = "쇼츠 영상 생성 중... 시간이 다소 걸릴 수 있습니다."

                composer = ShortsComposer(
                    luma_key=LUMA_KEY,
                    aws_key=AWS_ACCESS_KEY,
                    aws_secret=AWS_SECRET_KEY,
                    openai_key=OPENAI_KEY
                )

                ref_imgs = ["card.png"]
                video_path="shorts_kbank.mp4",
                audio_path="tts_kbank.mp3",
                srt_path="subs_kbank.srt",
                output_video_path="shorts_testtest.mp4"

                composer.run_all(
                    img_prompt=image_prompt,
                    input_images=ref_imgs,
                    user_request=subtitle_prompt,
                    video_path=video_path,
                    audio_path=audio_path,
                    srt_path=srt_path,
                    output_video_path=output_video_path
                )

                st.success("영상 생성 완료!")

        if st.session_state.result_video_bytes:
            st.subheader("📺 영상 미리보기")
            st.video(st.session_state.result_video_bytes, format="video/mp4")
            st.download_button(
                label="📥 영상 다운로드",
                data=st.session_state.result_video_bytes,
                file_name="shorts_video.mp4",
                mime="video/mp4"
            )