import json
import subprocess
import shutil
import base64
import re
import os
from google.genai import types
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def split2type(image_dir, curdir, category, app_name, image_name, content):
    IMG_LIMIT = 10
    image_files = []

    # Load complete image
    complete_img_path = os.path.join(image_dir, category, app_name, image_name)
    with open(complete_img_path, "rb") as f:
        complete_image_bytes = f.read()
    complete_img_part = types.Part.from_bytes(
        data=complete_image_bytes,
        mime_type="image/png"
    )

    # Load split images
    split_img_dir = os.path.join(curdir, "split_img", category, app_name, image_name.split(".")[0])
    png_files = sorted([f for f in os.listdir(split_img_dir) if f.endswith(".png")],
                       key=lambda x: int(os.path.splitext(x)[0]))

    for file in png_files:
        image_path = os.path.join(split_img_dir, file)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_files.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

    # Add text prompt as the last part
    image_files.append(types.Part.from_text(text=content))

    client = Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Call the API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=image_files
    )

    # Extract text
    generated_text = ""
    try:
        generated_text = response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
        generated_text = ""

    return generated_text


def extract_component_type(s):
    pattern = r'@@@(.*?)@@@'
    matches = re.findall(pattern, s)
    unique_matches = list(set(matches))  
    return unique_matches

def copy_file(src, dst):
    try:
        if not os.path.exists(src):
            print(f"Source file '{src}' does not exist.")
            return

        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            print(f"Destination directory '{dst_dir}' created.")

        command = ['cp', src, dst]
        subprocess.run(command, check=True)
        print(f"Successfully copied '{src}' to '{dst}'.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while copying: {e}")
        
def clear_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        
        print(f"Successfully cleared all contents of '{folder_path}'.")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    component_info_folder = "./results/extract_results"
    lang = "Flutter"
    image_dir = "./Dataset"

    category = "Entertainment"
    app = "musicAPP"

    initial_component_info_dir = os.path.join(component_info_folder, lang)
    initial_split_img_dir = os.path.join("./split_img")
    initial_complete_img_dir = image_dir

    prompt_dir = "./prompts"
    prompt_path = os.path.join(prompt_dir, "Flutter_Split2TypePrompt.txt")
    
    with open(prompt_path, "r") as f:
        split2type_prompt = f.read()
    
    # State File Creation
    state_file = {}
    complete_img_dir = os.path.join(initial_complete_img_dir, category)
    split_img_dir = os.path.join(initial_split_img_dir, category)
    component_info_dir = os.path.join(initial_component_info_dir, category)
    image_state = {}
    for image_name in os.listdir(os.path.join(complete_img_dir, app)):          
        if image_name.endswith('.png'):
            image_state[image_name.split('.')[0]] = 0
            state_file[app] = image_state
    if os.path.exists('state.json'):
        with open('state.json', 'r', encoding='utf-8') as f:
            state_file = json.load(f)

    print(f"Processing category: {category}")
    app_component_info_dir = os.path.join(component_info_dir, app)
    if not os.path.exists(app_component_info_dir):
        os.makedirs(app_component_info_dir)
    app_split_img_dir = os.path.join(split_img_dir, app)
    app_img_dir = os.path.join(complete_img_dir, app)
    for image_name in os.listdir(app_img_dir):
        if not image_name.endswith('.png'):
            continue
        if state_file[app][image_name.split('.')[0]] == 1:
            continue
        print(f"Processing {app} {image_name}") 
        component_type = split2type(image_dir, ".", category, app, image_name, split2type_prompt)
        component_type_path = os.path.join(app_component_info_dir, image_name.split('.')[0] + "_type.txt")
        with open(component_type_path, 'w', encoding='utf-8') as file:
            file.write(component_type)
        position_before = os.path.join(app_split_img_dir, image_name.split('.')[0], "position.txt")
        position_after = os.path.join(app_component_info_dir, image_name.split('.')[0] + "_position.txt")
        copy_file(position_before, position_after)
        state_file[app][image_name.split('.')[0]] = 1
        with open('state.json', 'w', encoding='utf-8') as f:
            json.dump(state_file, f, ensure_ascii=False, indent=4)
        print(f"Finished {app} {image_name}")

    print(f"Finished {app}")
    print(f"Finished {category}")