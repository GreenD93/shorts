import base64
import subprocess
import asyncio
from openai import OpenAI, AsyncOpenAI

class SubtitleAudioGenerator:
    TTS_PROMPT = """
        TTS로 말했을 때 총 발화 시간이 8.5초 이내여야 해.
        각 줄은 되도록 8자 이내, 짧고 리듬감 있게 만들어줘.
        말했을 때 자연스럽고 부드러운 흐름이 되도록 해줘.
        번호나 따옴표 없이, 줄바꿈으로만 구분해줘.
        
        쇼츠로 만드려고하는 이미지 썸네일도 같이 첨부 해줄게
        반드시 추가 설명 없이 자막 8줄만 만들어줘.
        
        케이뱅크에서 새로나온 One체크카드를 소개하는 쇼츠 영상 자막 8줄 만들어줘
        
        {user_request}
    """

    def __init__(self, openai_key):
        self.client = OpenAI(api_key=openai_key)
        self.async_client = AsyncOpenAI(api_key=openai_key)

    def generate_subtitles(self, user_request: str):
        
        prompt = self.TTS_PROMPT.format(user_request=user_request)

        content = [{"type": "input_text", "text": prompt}]

        response = self.client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": content}]
        )
        lines = [line.strip() for line in response.output_text.strip().split("\n") if line.strip()]
        return lines

    async def _generate_tts_file(self, text: str, index: int, voice="shimmer", model="tts-1"):
        filename = f"line_{index + 1}.mp3"
        try:
            response = await self.async_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ TTS 완료: {filename}")
            return filename
        except Exception as e:
            print(f"❌ TTS 오류 (line {index + 1}): {e}")
            return None

    async def _generate_all_tts_files(self, lines, voice="shimmer"):
        tasks = [
            self._generate_tts_file(line, idx, voice=voice)
            for idx, line in enumerate(lines)
        ]
        return await asyncio.gather(*tasks)

    def _merge_audio_files(self, tts_files: list, output_path: str):
        with open("concat_list.txt", "w") as f:
            for fname in tts_files:
                f.write(f"file '{fname}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt", "-c", "copy", output_path
        ], check=True)
        return output_path

    def generate_audio_and_subtitles(self, user_request: str, output_audio_path="combined.mp3"):
        
        print("[SubtitleAudio] 자막 생성 중...")
        subtitles = self.generate_subtitles(user_request=user_request)

        print("[SubtitleAudio] TTS 생성 중...")
    
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
    
        if loop and loop.is_running():
            # 이미 실행 중인 루프가 있는 경우 (예: Jupyter)
            print("[SubtitleAudio] 기존 이벤트 루프 사용 중 → asyncio.create_task 방식 사용 필요")
            # 임시 동기 처리 (강제적)
            import nest_asyncio
            nest_asyncio.apply()
            tts_files = asyncio.run(self._generate_all_tts_files(subtitles))
        else:
            tts_files = asyncio.run(self._generate_all_tts_files(subtitles))
    
        print("[SubtitleAudio] 오디오 병합 중...")
        audio_path = self._merge_audio_files(tts_files, output_path=output_audio_path)
    
        return audio_path, subtitles

    def generate_srt(self, lines: list, output="output.srt"):

        with open(output, "w", encoding="utf-8") as f:
            for i, line in enumerate(lines):
                start = f"00:00:0{i}.000"
                end = f"00:00:0{i + 1}.000"
                f.write(f"{i + 1}\n{start} --> {end}\n{line}\n\n")
        return output