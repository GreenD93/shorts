from luma_generator import LumaGenerator
from img_generator import ImgGenerator

from PIL import Image
import io

import base64
from openai import OpenAI

class VideoGenerator:

    LUMA_VIDEO_PROMPT = """
    You are a cinematic scene describer trained to generate short, vivid prompts for Luma Labs Dream Machine.
    
    Given a visual scene, generate a short cinematic prompt (15–35 words) that describes the subject, action, mood, lighting, and camera movement or angle.
    
    Examples:
    
    1. "A massive orb of water floating in a backlit forest"
    2. "An explosion with camera shake"
    3. "Raindrops in extreme slow motion"
    4. "A snow leopard crouched on a rocky ledge, staring directly at camera, snowflakes falling around it"
    5. "A detective in a dark trench coat and hat, holding a lantern, walks carefully down a narrow cobblestone alley, mist curling around his feet"
    6. "At sunrise, place the camera at ground level, looking up at a solitary warrior standing on a rugged cliff edge, wind pulling at his cloak"
    7. "A tiny chihuahua dressed in post-apocalyptic leathers with a WW2 bomber cap and goggles sits proudly on a makeshift throne"
    8. "A cellist alone on stage under a single spotlight, camera slowly circling her as she begins to play"
    9. "A gorilla surfing on a wave, water splashing dramatically as it leans into the motion"
    
    Now, based on the input image or concept, generate a similarly styled prompt that is cinematic, physical, and emotionally engaging.
    
    Output only the prompt text.
    
    """

    def __init__(self, luma_key, aws_key, aws_secret, openai_key):
        self.luma = LumaGenerator()
        self.luma.set_luma_client(luma_key)
        self.luma.set_s3_client(aws_key, aws_secret)

        self.img = ImgGenerator()
        self.img.set_openai_key(openai_key)
        self.openai = OpenAI(api_key=openai_key)

    def generate_image(self, prompt, input_images=None, img_path="temp123456789.png"):

        if input_images:
            
            print(prompt, input_images)
            
            imgs, _ = self.img.edit_multiple_images(
                prompt=prompt,
                images=input_images,
                n=1,
                size="1024x1024",
                background="opaque",
                quality="high",
                model="gpt-image-1",
                timer_interval=2
            )

        else:
            
            imgs, _ = self.img.generate_multiple_images(
                prompt=prompt,
                n=1,
                size="1024x1024",
                background="opaque",
                quality="high",
                model="gpt-image-1",
                timer_interval=2
            )

        image_bytes = imgs[0] # bytes
        
        # image_bytes → PNG 파일로 저장
        image = Image.open(io.BytesIO(image_bytes))  # ✅ 여기에 image_bytes
        image.save(img_path, format="PNG")           # ex. img_path = "temp.png"
        
        return image_bytes, img_path

    def generate_video_from_image(self, cinematic_prompt, img_path):
        
        video_bytes, _ = self.luma.generate_video(
            prompt=cinematic_prompt,
            keyframes=[img_path],
            loop=True,
            model="ray-2",
            aspect_ratio="9:16",
            resolution="720p",
            duration="9s"
        )
        
        return video_bytes

    def _get_cinematic_prompt(self, image_bytes):

        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        
        response = self.openai.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self.LUMA_VIDEO_PROMPT},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_img}"},
                    ],
                }
            ]
        )
        return response.output_text

    def generate_video_from_prompt(self, image_prompt: str, 
                                   input_images: list = [],
                                   output_path: str = "shorts.mp4") -> bytes:
        
        print("[VideoGen] 이미지 생성 중...")
        image_bytes, img_path = self.generate_image(prompt=image_prompt, input_images=input_images)

        cinematic_prompt = self._get_cinematic_prompt(image_bytes)

        print(cinematic_prompt)
        
        print("[VideoGen] 비디오 생성 중...")
        video_bytes = self.generate_video_from_image(cinematic_prompt, img_path)

        print(f"[VideoGen] 비디오 저장: {output_path}")
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        return video_bytes