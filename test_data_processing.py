import sys
from unittest.mock import MagicMock

# Mocking libraries that are not installed or too heavy
sys.modules['unsloth'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()

import dataset_utils_GRPO
from PIL import Image
import os

def test_tiling():
    # Create two test images
    img1 = Image.new('RGB', (100, 50), (255, 0, 0))
    img2 = Image.new('RGB', (100, 50), (0, 255, 0))

    images = [img1, img2]
    tiled = dataset_utils_GRPO.tile_images(images)

    print(f"Tiled image size: {tiled.size}")
    assert tiled.size == (100, 100)
    print("Tiling size test passed!")

def test_transformation():
    # Ensure images exist
    os.makedirs('gemma3_dataset_GRPO', exist_ok=True)
    Image.new('RGB', (100, 100), (255, 0, 0)).save('gemma3_dataset_GRPO/image1.png')
    Image.new('RGB', (100, 100), (0, 255, 0)).save('gemma3_dataset_GRPO/image2.png')

    data = {
        "instruction": ["Inst1"],
        "input": ["Input1"],
        "output": ["Output1"],
        "images": [["gemma3_dataset_GRPO/image1.png", "gemma3_dataset_GRPO/image2.png"]]
    }

    result = dataset_utils_GRPO.final_unsloth_transformation(data)

    prompt = result['prompt'][0]
    content = prompt[0]['content']

    # Check number of image placeholders
    image_placeholders = [c for c in content if c['type'] == 'image']
    print(f"Number of image placeholders: {len(image_placeholders)}")
    assert len(image_placeholders) == 1

    # Check number of images in final_images
    print(f"Number of images in result: {len(result['images'][0])}")
    assert len(result['images'][0]) == 1

    print("Transformation test passed!")

if __name__ == "__main__":
    try:
        test_tiling()
        test_transformation()
        print("\nALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
