import json
import os
import base64
from google.genai import types
from google.genai import Client
import httpx
from dotenv import load_dotenv

load_dotenv()

picture_folder = "./Dataset"
ptg_prompt_path = "./PTG_Prompt.txt"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ptg_generate(image_dir, category, app_name, content):
    image_files = []
    app_img_folder = os.path.join(image_dir, category, app_name)

    files = os.listdir(app_img_folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_file_names = [f.split('.')[0] for f in png_files]
    content = content.format(node_id_list=png_file_names)

    client = Client()

    # Prepare image parts
    for file in png_files:
        image_path = os.path.join(app_img_folder, file)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_files.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )
            )

    # Add the text prompt as the final part
    image_files.append(content)

    # Call the API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=image_files
    )

    # Extract generated text
    generated_text = ""
    try:
        generated_text = response.candidates[0].content.parts[0].text.strip()
        generated_text = generated_text.replace('```json\n','').replace('```','')
    except Exception as e:
        print(f"Error parsing response: {e}")
        generated_text = "{}"

    return json.loads(generated_text)

if __name__ == "__main__":
    category = "Entertainment"
    app_name = "musicAPP"
    with open(ptg_prompt_path, 'r', encoding='utf-8') as f:
        ptg_prompt = f.read()
    res = ptg_generate(picture_folder, category, app_name, ptg_prompt)
    with open('./ptg.json', 'w+') as f:
        json.dump(res, f, indent=4)
    print(f"PTG Output for {app_name} Saved!")