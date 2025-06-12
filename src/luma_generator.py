import io
import time
import boto3
import base64

import asyncio
import requests

from PIL import Image
from botocore.exceptions import NoCredentialsError

from lumaai import AsyncLumaAI
from lumaai import LumaAI

from typing import Optional, List, Union

from IPython.display import display

from rembg import remove

# 설정
GENERATION_POLL_INTERVAL = 5  # seconds

class LumaGenerator:

    def __init__(self):
        self.luma_client: Optional[LumaAI] = None
        self.luma_async_client: Optional[LumaAI] = None
        self.s3_client: Optional[boto3.client] = None

        self.prompt_helper: Optional[PromptHelper] = None

    def set_luma_client(self, luma_key: str) -> None:
        self.luma_client = LumaAI(auth_token=luma_key)
        self.luma_async_client = AsyncLumaAI(auth_token=luma_key)
        

    def set_s3_client(self, aws_access_key: str, aws_secret_key: str,
                            region_name: str = "ap-northeast-2") -> None:

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )

    def set_prompt_helper(self, prompt_helper: "PromptHelper"):
        self.prompt_helper = prompt_helper
        
    def upload_file(self, file_path: str, s3_path: str,
                    bucket_name: str = "kbank-data-intelligence", region_name: str = "ap-northeast-2") -> Optional[str]:
        try:
            self.s3_client.upload_file(file_path, bucket_name, s3_path)
            file_url = f'https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_path}'
            print(f'File uploaded successfully. Public URL: {file_url}')
            return file_url
        except NoCredentialsError:
            print("Credentials not available")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

    def get_image_bytes(self, generation) -> bytes:
        image_url = generation.assets.image
        response = requests.get(image_url, stream=True)
        return response.content

    def save_image(self, img_bytes: bytes, img_path: str) -> None:
        with open(f'{img_path}', 'wb') as file:
            file.write(img_bytes)
        print(f"File saved as {img_path}.png")
    
    def show_image(self, *img_bytes_list: Union[bytes, list]) -> None:
        """
        최대 4개의 이미지 바이트를 인자로 받아 Jupyter에서 일렬로 표시합니다.
    
        사용 예:
            self.show_image(img1)
            self.show_image(img1, img2, img3)
        """
        from IPython.display import display, HTML
    
        # img_bytes_list가 리스트로 들어온 경우 평탄화
        images = list(img_bytes_list[0]) if len(img_bytes_list) == 1 and isinstance(img_bytes_list[0], list) else list(img_bytes_list)
        images = images[:4]  # 최대 4개 제한
    
        html = "<div style='display:flex; gap:10px;'>"
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            html += f"<img src='data:image/png;base64,{b64}' width='200px'/>"
        html += "</div>"
    
        display(HTML(html))

    def remove_bg(self, img_bytes):
        out = remove(img_bytes)
        return out


    async def _generate_single_image(self, prompt: str, model:str,
                                    image_refs: Optional[List[dict]] = None,
                                    style_refs: Optional[List[str]] = None,
                                    modify_refs: Optional[List[str]] = None,
                                    aspect_ratio: Optional[str] = None,
                                    max_retries: int = 1) -> dict:

        attempt = 0
        
        while attempt <= max_retries:
            try:
                print(f"[↺] Attempt {attempt + 1} for image generation")
                params = {"model": model, "prompt": prompt}
                if image_refs:
                    params["image_ref"] = image_refs
                if style_refs:
                    params["style_ref"] = style_refs
                if modify_refs:
                    params["modify_refs"] = modify_refs
                if aspect_ratio:
                    params["aspect_ratio"] = aspect_ratio
    
                generation = await self.luma_async_client.generations.image.create(**params)
    
                while generation.state not in ["completed", "failed"]:
                    await asyncio.sleep(GENERATION_POLL_INTERVAL)
                    generation = await self.luma_async_client.generations.get(id=generation.id)
    
                if generation.state == "completed":
                    print(f"[✔] Completed image: {generation.id}")
                    return generation
                else:
                    print(f"[✖] Failed image: {generation.id} - {generation.failure_reason}")
            except Exception as e:
                print(f"[!] Exception during async image generation (Attempt {attempt + 1}): {str(e)}")
    
            attempt += 1
            if attempt <= max_retries:
                await asyncio.sleep(2)  # 재시도 간 짧은 대기 시간
    
        print("[❌] All attempts failed for image generation.")
        
        return None

    async def _generate_multiple_images_async(self, prompt: str, model:str, n: int = 4,
                                              image_refs: Optional[List[dict]] = None,
                                              style_refs: Optional[List[str]] = None,
                                              modify_refs: Optional[List[str]] = None,
                                              aspect_ratio: Optional[str] = None,
                                              ) -> List[dict]:

        start_time = time.time()  # 시작 시간 측정
        
        tasks = [
            self._generate_single_image(
                model=model,
                prompt=prompt,
                image_refs=image_refs,
                style_refs=style_refs,
                modify_refs=modify_refs,
                aspect_ratio=aspect_ratio
            ) for _ in range(n)
        ]
        results = await asyncio.gather(*tasks)

        end_time = time.time()  # 종료 시간 측정
        total_duration = end_time - start_time
        print(f"[⏱] All async generations completed in {total_duration:.2f} seconds")

        return [res for res in results if res is not None]

    def generate_multiple_images(self, prompt: str, model: str = "photon-1", n: int = 4,
                                 image_refs: Optional[List[dict]] = None,
                                 style_refs: Optional[List[str]] = None,
                                 modify_refs: Optional[List[str]] = None,
                                 aspect_ratio: Optional[str] = None,
                                 s3_prefix: str = "kpick/img",
                                 use_prompt_helper: bool=False) -> list:

        """
        image_refs: List of dicts like {"path": "local/path.jpg", "weight": 0.8}
        s3_prefix: S3 업로드 경로 prefix (ex. "kpick/img", "temp/test", etc.)
        """

        if n > 4:
            raise ValueError("최대 4개까지만 생성 가능합니다. n <= 4로 설정해주세요.")
            
        # loop = asyncio.get_event_loop() # Streamlit은 자체적으로 비동기 루프를 사용하지 않는 스레드에서 코드를 실행합니다.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def process_refs(refs: Optional[List[dict]]) -> Optional[List[dict]]:
            if not refs:
                return None

            processed = []
            for ref in refs:
                path = ref.get("path")
                weight = ref.get("weight", 1.0)

                if not path:
                    raise ValueError("Each image_ref must contain 'path' key.")

                if path.startswith("http://") or path.startswith("https://"):
                    url = path
                else:
                    filename = path.split("/")[-1]
                    s3_path = f"{s3_prefix}/{filename}"
                    url = self.upload_file(path, s3_path)
                    if not url:
                        raise RuntimeError(f"Failed to upload: {path}")

                processed.append({"url": url, "weight": weight})

            return processed

        if image_refs:
            image_refs = process_refs(image_refs)
        if style_refs:
            style_refs = process_refs(style_refs)
        if modify_refs:
            modify_refs = process_refs(modify_refs)

        refs = image_refs or style_refs or modify_refs
        
        if use_prompt_helper:
            prompt = self.prompt_helper.rewrite(prompt=prompt, paths=refs, mode="img_prompt")

        results = loop.run_until_complete(
                    self._generate_multiple_images_async(
                                                          prompt=prompt,
                                                          model=model,
                                                          n=n,
                                                          image_refs=image_refs,
                                                          style_refs=style_refs,
                                                          modify_refs=modify_refs,
                                                          aspect_ratio=aspect_ratio
                                                        )
                    )

        return results, prompt
    
    def _generate_video(self,
                        prompt,
                        keyframes,
                        loop,
                        model,
                        aspect_ratio,
                        resolution,
                        duration):
        
        generation = self.luma_client.generations.create(
                                              prompt=prompt,
                                              keyframes=keyframes,
                                              loop=loop,
                                              model=model,
                                              aspect_ratio=aspect_ratio,
                                              resolution=resolution,
                                              duration=duration
                                            )
        
        return generation
    
    def generate_video(self, prompt,
                       keyframes=None,
                       loop=None,
                       model="ray-flash-2",
                       aspect_ratio="4:3",
                       resolution="720p",
                       duration="5s",
                       s3_prefix: str = "kpick/img",
                       timer_interval: int = 3,
                       use_prompt_helper: bool=False
                      ):

        def process_keyframes(keyframes):

            if not keyframes:
                return None

            processed = {}

            for idx, (key, value) in enumerate(keyframes.items()):

                path = value.get("path")

                if not path:
                    raise ValueError("Each image_ref must contain 'path' key.")

                if path.startswith("http://") or path.startswith("https://"):
                    url = path

                else:
                    filename = path.split("/")[-1]
                    s3_path = f"{s3_prefix}/{filename}"
                    url = self.upload_file(path, s3_path)
                    if not url:
                        raise RuntimeError(f"Failed to upload: {path}")

                processed[f"frame{idx}"] = {
                                            "type": "image",
                                            "url": url
                                            }

            return processed

        if use_prompt_helper:
            prompt = self.prompt_helper.rewrite(prompt=prompt, paths=keyframes, mode="luma_video_prompt")

        keyframes = {f"frame{idx}": {"type": "image", "path": keyframe} for idx, keyframe in enumerate(keyframes)} if keyframes else None
        keyframes = process_keyframes(keyframes)

        generation = self._generate_video(
                                              prompt=prompt,
                                              keyframes=keyframes,
                                              loop=loop,
                                              model=model,
                                              aspect_ratio=aspect_ratio,
                                              resolution=resolution,
                                              duration=duration
                                            )
        
        completed = False
        last_elapsed = -1
        start_time = time.time()

        while not completed:

            elapsed = int(time.time() - start_time)

            if elapsed != last_elapsed and elapsed % timer_interval == 0:

                print(f"\r 비디오 생성 중... (Elapsed: {elapsed}초)", end='', flush=True)

                generation = self.luma_client.generations.get(id=generation.id)

                if generation.state == "completed":
                    completed = True

                elif generation.state == "failed":
                    raise RuntimeError(f"Generation failed: {generation.failure_reason}")

                last_elapsed = elapsed
            
            time.sleep(0.5)  # 너무 자주 체크하지 않게 약간 딜레이

        video_url = generation.assets.video
        response = requests.get(video_url, stream=True)
        video_bytes = response.content
        
        return video_bytes, prompt