import base64
import time
import threading
from openai import OpenAI

from typing import Optional, List, Union
from IPython.display import display, HTML
from tqdm import tqdm

from rembg import remove

class ImgGenerator:
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._running: bool = False  # 시간 출력 제어용 플래그

        self.prompt_helper: Optional[PromptHelper] = None

    def set_openai_key(self, openai_key: str) -> None:
        self.client = OpenAI(api_key=openai_key)

    def set_prompt_helper(self, prompt_helper: "PromptHelper"):
        self.prompt_helper = prompt_helper

    def _timer_thread(self, n: int, interval: int = 2):
        """
        API 호출 대기 중 경과 시간 출력용 쓰레드

        Args:
            n (int): 생성할 이미지 개수
            interval (int): 경과 시간 출력 간격 (초)
        """
        start_time = time.time()
        last_elapsed = -1
        while self._running:
            elapsed = int(time.time() - start_time)
            if elapsed != last_elapsed and elapsed % interval == 0:
                print(f"\r이미지 {n}개 생성 중... (Elapsed: {elapsed}초)", end='', flush=True)
                last_elapsed = elapsed
            time.sleep(0.5)  # 너무 자주 체크하지 않게 약간 딜레이

    def get_image_bytes(self, response) -> bytes:
        image_base64 = response.b64_json
        image_bytes = base64.b64decode(image_base64)
        return image_bytes

    def generate_multiple_images(self, prompt: str, n: int=1,
                                  size: str="1024x1024", background: str="transparent",
                                  quality: str="auto", model: str="gpt-image-1",
                                  timer_interval: int=2,
                                  use_prompt_helper: bool=False) -> List[bytes]:
        """
        프롬프트를 기반으로 여러 이미지를 생성하는 함수.
    
        Args:
            prompt (str): 생성할 이미지에 대한 텍스트 프롬프트
            n (int, optional): 생성할 이미지 수. 기본값은 1
            size (str, optional): 생성할 이미지 크기. 기본값은 "1024x1024"
            background (str, optional): 배경 설정. 기본값은 "transparent"
            quality (str, optional): 이미지 품질 설정. 기본값은 "auto"
            model (str, optional): 사용할 OpenAI 모델명. 기본 "gpt-image-1"
            timer_interval (int, optional): 진행상황 경과시간 출력 주기 (초). 기본 2초
    
        Returns:
            List[bytes]: 생성된 이미지의 바이트 리스트
        """
        
        self._running = True
        timer = threading.Thread(target=self._timer_thread, args=(n, timer_interval))
        timer.start()

        start_time = time.time()
        
        if use_prompt_helper:
            prompt = self.prompt_helper.rewrite(prompt=prompt, mode="img_prompt")

        try:
            response = self._generate_multiple_images(
                prompt=prompt, n=n, size=size, background=background, quality=quality, model=model
            )
        finally:
            self._running = False
            timer.join()

        elapsed_time = time.time() - start_time
        
        print(f"\n[API 호출 완료] 총 소요 시간: {elapsed_time:.2f}초")

        imgs = []
        for data in tqdm(response.data, desc="이미지 변환 중", unit="img"):
            img_bytes = self.get_image_bytes(data)
            imgs.append(img_bytes)

        return imgs, prompt

    def _generate_multiple_images(self, prompt: str, n: int,
                                   size: str, background: str, quality: str,
                                   model: str):
        """
        내부용 API 호출 함수.
        """
        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            background=background,
            quality=quality
        )
        
        return response

    def edit_multiple_images(self, prompt: str, images: list, n: int=1,
                                  size:str ="1024x1024", background: str="transparent",
                                  quality: str="auto", model: str="gpt-image-1",
                                  timer_interval: int=2,
                                  use_prompt_helper: bool=False) -> List[bytes]:
        """
        프롬프트를 기반으로 여러 이미지를 생성하는 함수.
    
        Args:
            prompt (str): 생성할 이미지에 대한 텍스트 프롬프트
            images (list): 편집할 이미지에 대한 경로
            n (int, optional): 생성할 이미지 수. 기본값은 1
            size (str, optional): 생성할 이미지 크기. 기본값은 "1024x1024"
            background (str, optional): 배경 설정. 기본값은 "transparent"
            quality (str, optional): 이미지 품질 설정. 기본값은 "auto"
            model (str, optional): 사용할 OpenAI 모델명. 기본 "gpt-image-1"
            timer_interval (int, optional): 진행상황 경과시간 출력 주기 (초). 기본 2초
    
        Returns:
            List[bytes]: 생성된 이미지의 바이트 리스트
        """
        
        self._running = True
        timer = threading.Thread(target=self._timer_thread, args=(n, timer_interval))
        timer.start()

        start_time = time.time()
        
        if use_prompt_helper:
            prompt = self.prompt_helper.rewrite(prompt=prompt, paths=images, mode="img_prompt")

        try:
            response = self._edit_multiple_images(
                            prompt=prompt, images=images, n=n, 
                            size=size, quality=quality, model=model
                        )
        finally:
            self._running = False
            timer.join()

        elapsed_time = time.time() - start_time
        
        print(f"\n[API 호출 완료] 총 소요 시간: {elapsed_time:.2f}초")

        imgs = []
        
        for data in tqdm(response.data, desc="이미지 변환 중", unit="img"):
            
            img_bytes = self.get_image_bytes(data)

            if background == "transparent": # 배경 제거
                img_bytes = self.remove_bg(img_bytes)
            
            imgs.append(img_bytes)

        return imgs, prompt

    def _edit_multiple_images(self, prompt: str, images: list, n: int,
                                   size: str, quality: str, model: str):

        images = [open(image, "rb") for image in images]
        
        response = self.client.images.edit(
            model=model,
            image=images,
            prompt=prompt,
            n=n,
            size=size,
            quality=quality
        )

        return response
        
    def show_image(self, *img_bytes_list: Union[bytes, List[bytes]],
                         n=4) -> None:

        """
        최대 N개의 이미지 바이트를 받아서 Jupyter에서 표시합니다.
        """

        images = list(img_bytes_list[0]) if len(img_bytes_list) == 1 and isinstance(img_bytes_list[0], list) else list(img_bytes_list)
        images = images[:n]  # 최대 N개 제한

        html = "<div style='display:flex; gap:10px;'>"
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            html += f"<img src='data:image/png;base64,{b64}' width='200px'/>"
        html += "</div>"

        display(HTML(html))

    def save_image(self, img_bytes: bytes, img_path: str) -> None:
        """
        이미지를 로컬 파일로 저장합니다.
        """
        if not (img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
            img_path += '.png'
        with open(img_path, 'wb') as file:
            file.write(img_bytes)
        print(f"File saved as {img_path}")

    def remove_bg(self, img_bytes):
        out = remove(img_bytes)
        return out