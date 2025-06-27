import subprocess
from threading import Thread

from src.video_generator import VideoGenerator
from src.subtitle_generator import SubtitleAudioGenerator

class ShortsComposer:

    def __init__(self, luma_key, aws_key, aws_secret, openai_key):

        self.video_gen = VideoGenerator(luma_key, aws_key, aws_secret, openai_key)
        self.subtitle_gen = SubtitleAudioGenerator(openai_key)

        self.video_bytes = None
        self.audio_path = None
        self.subtitles = None

    def _run_video_thread(self, img_prompt: str, input_images: list, video_path: str):
        print("[🎞️ Thread: Video] 영상 생성 시작")
        
        self.video_bytes = self.video_gen.generate_video_from_prompt(
            image_prompt=img_prompt,
            input_images=input_images,
            output_path=video_path
        )
        print("[🎞️ Thread: Video] 영상 생성 완료")

    def _run_audio_thread(self, user_request: str, audio_path: str):

        print("[🎤 Thread: Audio] 오디오+자막 생성 시작")

        self.audio_path, self.subtitles = self.subtitle_gen.generate_audio_and_subtitles(
            user_request=user_request,
            output_audio_path=audio_path
        )

        print("[🎤 Thread: Audio] 오디오+자막 생성 완료")

    def run_all(self,
                img_prompt: str,
                input_images: list,
                user_request: str,
                video_path: str = "shorts.mp4",
                audio_path: str = "combined.mp3",
                srt_path: str = "output.srt",
                output_video_path: str = "final_output.mp4"):

        print("🚀 쇼츠 생성 시작")

        self._run_video_thread(img_prompt, input_images, video_path)
        self._run_audio_thread(user_request, audio_path)

        # 병렬 처리
        # thread_video = Thread(target=self._run_video_thread, args=(img_prompt, input_images, video_path))
        # thread_audio = Thread(target=self._run_audio_thread, args=(user_request, audio_path))

        # thread_video.start()
        # thread_audio.start()

        # thread_video.join()
        # thread_audio.join()

        print("📝 자막 파일 생성 중...")
        print(self.subtitles)

        self.subtitle_gen.generate_srt(self.subtitles, output=srt_path)

        print("🎬 최종 비디오 병합 중...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-vf", f"subtitles={srt_path}",
            "-c:v", "libx264", "-c:a", "aac",
            output_video_path
        ], check=True)

        print(f"✅ 쇼츠 생성 완료: {output_video_path}")