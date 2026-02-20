import json
import re
import random
from PIL import Image
import os
from datasets import Dataset, Features, Sequence, Value, Image as datasets_Image

_tokenizer = None

dummy_image_path = "gemma3_dataset_GRPO/dummy_blank.png"
if not os.path.exists(dummy_image_path):
    os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
    Image.new("RGB", (256, 256), (255, 255, 255)).save(dummy_image_path)


def tile_images(images):
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    # Vertical tiling
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    return combined_image


def generate_documentation_string(doc, args):
    dataset = "PARTIAL TECHNICAL SCHEMA REFERENCE:"

    dataset += f"\n--- Variables ---\n"

    for category in doc.get("VariableCategories", []):
        # cat_name = category.get("Category", "Unknown")
        # dataset += f"\n-- {cat_name} --\n"

        for variable in category.get("Variables", []):
            name = variable.get("Name", "")

            if name in args or random.random() <= 0.025:
                v_type = variable.get("Type", None)
                val = variable.get("Value", None)
                dataset += f"{name}"

                if v_type is not None:
                    dataset += f" ({v_type})"

                if val is not None:
                    dataset += f": {val}"

                dataset += f";"

                values_list = variable.get("Values")
                if values_list is not None:
                    dataset += f"Available values ["

                    for value in values_list:
                        v_id = value.get("Value", "?")
                        v_name = value.get("Name", "Unnamed")
                        dataset += f"{v_id}: ({v_name});"

                    dataset += f"]"

    dataset += f"\n--- Operators ---\n"

    for operator in doc.get("Operators", []):
        name = operator.get("Name", "N/A")
        desc = operator.get("Description", "N/A")
        ex = operator.get("Example", "N/A")
        # dataset += f"{name};"
        dataset += f"{name} ({desc}) example: {ex};"

    dataset += f"\n--- Functions ---\n"

    for function in doc.get("Functions", []):
        name = function.get("Name", "N/A")
        desc = function.get("Description", "N/A")
        ex = function.get("Example", "N/A")
        # dataset += f"{name};"
        dataset += f"{name} ({desc}) example: {ex};"

    # dataset += f"\n--- Example pricing rule formulas ---\n"

    # for formula in doc.get("Examples", []):
    #     dataset += f"formula\n"

    return dataset


def final_unsloth_transformation(data):
    instructions = data.get("instruction", "")
    inputs = data.get("input", "")
    outputs = data.get("output","")  # This becomes the 'solution'
    image_paths = data.get("images", [])

    final_prompts = []
    final_images = []
    final_solutions = []

    for inst, inp, out, paths in zip(instructions, inputs, outputs, image_paths):
        row_images = []
        user_content = []

        # Sanitize prompt inputs to avoid empty inputs
        if not inp or inp.strip() == "":
            inp = "Analyze the image and provide the pricing rule."

        # Ensure 'paths' is a list (even if it's a single string)
        if isinstance(paths, str):
            paths = [paths]

        # 1. Loop through every path in the list for THIS prompt
        for p in paths:
            if not p or not os.path.exists(p):
                p = dummy_image_path

            img = Image.open(p).convert("RGB")
            row_images.append(img)

        # Tiling logic to prevent mismatch error in GRPO trainer
        if len(row_images) > 1:
            row_images = [tile_images(row_images)]

        # Add placeholders (one per image in row_images)
        for _ in range(len(row_images)):
            # This creates the exact placeholder Qwen-VL needs
            user_content.append({"type": "image"})

        # 2. Add the Reasoning Instruction
        # We tell the model HOW to think so the Reward Function can find the tags
        reasoning_prompt = (
            """You are an expert at extracting pricing logic from screenshots or text.
            1. Analyze the inputs to find pricing tables and rules.
            2. Output your reasoning process inside <think> tags.
            3. Output the final pricing formula inside <answer> tags.
            4. Be concise.
            5. You can use the variables, operators and functions provided in the partial documentation to help you (not comprehensive).
            6. Do not use markdown for the formula, just raw text.""" f"{inst}\n{inp}\n"
        )
        user_content.append({"type": "text", "text": reasoning_prompt})

        # 3. GRPO Format: 'prompt' is JUST the user message
        prompt = [{"role": "user", "content": user_content}]

        final_prompts.append(prompt)
        final_images.append(row_images)  # Unsloth wants a list of images
        final_solutions.append(out)  # The raw ground truth formula

    return {
        "prompt": final_prompts,
        "images": final_images,
        "solution": final_solutions,
    }

def generate_dataset_string(data, base_dir):
    instructions = data.get("Instructions", [])

    # The Formatter (Generates the JSON structure above)
    final_dataset = []

    for item in instructions:
        # Your Static Documentation
        documentation = generate_documentation_string(
            data.get("Documentation", None),
            re.findall(r"{\w+}", item.get("Answer", "")),
        )

        prompt = item.get("Prompt", "")

        # 1. Get the list of image filenames from your source item
        # Defaults to an empty list if "Images" is missing
        image_filenames = item.get("Images", [])

        # 2. Build the full paths for all images in the list
        if isinstance(image_filenames, list) and len(image_filenames) > 0:
            full_paths = [os.path.join(base_dir, img) for img in image_filenames]
        else:
            # If the list is empty, fall back to your dummy image
            full_paths = [dummy_image_path]

        # 3. Create the full conversation object
        entry = {
            "instruction": f"{documentation}\n\nGENERATE PRICING FORMULAS.",
            "input": prompt,
            "output": item.get("Answer", ""),
            "images": full_paths,
        }

        final_dataset.append(entry)

    dataset_obj = Dataset.from_list(final_dataset)

    final_dataset = dataset_obj.map(
        final_unsloth_transformation,
        batched=True,
        remove_columns=dataset_obj.column_names,
        features=Features(
            {
                # Wrap the dictionary in [] to signify a list of messages
                "prompt": [
                    {
                        "role": Value("string"),
                        "content": [
                            {
                                "type": Value("string"),
                                "text": Value("string"),
                                # "image": datasets_Image(decode=False),
                            }
                        ],
                    }
                ],
                "solution": Value("string"),
                "images": Sequence(datasets_Image(decode=True)),
            }
        ),
    )

    return final_dataset


def get_valid_symbols(data):
    doc = data.get("Documentation", {})

    args = set()
    for category in doc.get("VariableCategories", []):
        for variable in category.get("Variables", []):
            name = variable.get("Name")
            if name:
                args.add(name)

    funcs = set()
    for func in doc.get("Functions", []):
        name = func.get("Name")
        if name:
            funcs.add(name)

    ops = set()
    for func in doc.get("Operators", []):
        name = func.get("Name")
        if name:
            ops.add(name)

    return args, funcs, ops
