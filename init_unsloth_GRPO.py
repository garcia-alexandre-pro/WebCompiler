import os
import sys
import shutil # Added to delete the cache

# --- STEP 0: NUCLEAR CACHE CLEAN (CRITICAL) ---
# The error proves the compiled kernel is still running. We must delete it physically.
cache_path = r"C:\Users\Admin\Downloads\LLMs\unsloth_compiled_cache"
if os.path.exists(cache_path):
    print(f"--- DELETING CORRUPTED CACHE: {cache_path} ---")
    try:
        shutil.rmtree(cache_path)
        print("Cache deleted successfully. This forces a clean, standard attention path.")
    except Exception as e:
        print(f"WARNING: Could not delete cache: {e}")

# --- HARDWARE & MEMORY OPTIMIZATION ---
os.environ["UNSLOTH_USE_COMPILED_KERNELS"] = "1" # 0 to prevent rebuild
os.environ["TORCH_COMPILE_DISABLE"] = "0" # 1 to prevent compilation
os.environ["UNSLOTH_USE_PEFT_WEIGHT_SHARING"] = "1" # Fix for 32GB memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GRPO & Unsloth Specifics
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force Single-GPU Mode (Critical for Windows TRL)
# Force Single-GPU Mode
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Force Single-Process Mode (Must be before any torch imports)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12445"

# Disable all distributed backends
os.environ["ACCELERATE_USE_CPU"] = "FALSE"
os.environ["ACCELERATE_USE_FSDP"] = "FALSE"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "FALSE"

# Explicitly tell Lightning/HF to use GLOO just in case it initializes
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import torch

torch._dynamo.config.suppress_errors = True

import json
import re
from dataset_utils_GRPO import generate_dataset_string, get_valid_symbols
from unsloth import FastVisionModel, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from difflib import SequenceMatcher
import gc

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

# 1. Configuration for RTX 5090
# model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit" # The Stable Choice
model_name = "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit" # The Stable Choice
# model_name = "unsloth/gemma-3-27b-it" # Remplacer par Gemma 3 Vision si dispo
max_seq_length = 8192 # Qwen2.5-VL handles context very efficiently
dtype = None # Auto-détection (bfloat16 pour Blackwell)
load_in_4bit = True # 4-bit quantization pour économiser la VRAM
seed = 3407
lora_rank = 128 # We can afford 16 on a 5090 with Qwen

# Blackwell/5090 works best with TF32 enabled for the non-BF16 parts
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Disable profiling and extra junk
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

print(f"--- Loading {model_name} on {torch.cuda.get_device_name(0)} ---")

# 2. Loading the Model and the Tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth", # Critical memory optimization
    max_lora_rank = lora_rank,
    fast_inference = False, # True if optimized (vllm only), but should disable finetune_vision_layers
    gpu_memory_utilization = 0.95, # 0.95 reduce if out of memory
    attn_implementation = "sdpa", # Standard PyTorch attention
    unsloth_force_compile = False,
    torch_dtype = torch.bfloat16,
    dtype = torch.bfloat16 # Use torch.float16, torch.bfloat16
)

# Shrink the vision tower's internal footprint
if hasattr(model, "vision_tower"):
    # model.vision_tower.config.patch_size = 14
    model.vision_tower.config.use_cache = True # Required for generation phase

# model.config.vision_config.image_size = 512
# model.config.vision_config.max_image_tiles = 1 # Stop VRAM-hungry tiling

# Explicitly cap pixel limits to prevent "high-res" token explosion
model.config.vision_config.min_pixels = 256 * 256
model.config.vision_config.max_pixels = 2048 * 2048 # Increased for tiled images

model.config.attn_logit_softcapping = None
model.config.final_logit_softcapping = None

# 3. Ajout des adaptateurs LoRA (Fine-Tuning efficace)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = False, # Important
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = lora_rank, # Rank (higher = smarter, max 64 recommended)
    lora_alpha = 2 * lora_rank,
    lora_dropout = 0,
    bias = "none",
    random_state = seed,
    use_rslora = True,
    loftq_config = None,
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # Qwen uses all these projections. Targeting them makes it smarter.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

# 4. Chargement du Dataset JSON
with open('gemma3_dataset_GRPO/raw_dataset.json') as file:
    raw_dataset = json.load(file)

dataset = generate_dataset_string(raw_dataset, "C:/Users/Admin/Downloads/LLMs/gemma3_dataset_GRPO/")
args, funcs, ops = get_valid_symbols(raw_dataset)

# # Format prompt for Gemma 3 Vision and map arguments correctly
# def format_for_vision(example):
#     user_content = []

#     for img in example["images"]:
#         user_content.append({
#             "type": "image",
#             "image": img
#         })

#     user_content.append({
#         "type": "text",
#         "text": example["prompt"]
#     })

#     return {
#         "prompt": [
#             {
#                 "role": "user",
#                 "content": user_content
#             }
#         ],
#         "answer": example['solution'],
#         "args": list(args),
#         "funcs": list(funcs),
#         "ops": list(ops)
#     }

# dataset = dataset.map(format_for_vision)

dataset = dataset.map(lambda x: {
    "args": list(args),
    "funcs": list(funcs),
    "ops": list(ops)
})

# 5. Configuration de l'Entraînement
training_args = GRPOConfig(
    output_dir = "outputs",

    # --- Generation & RL Params (The "Heart" of GRPO) ---
    num_generations = 8, # Sample 8 paths per prompt for the 5090 to compare
    max_prompt_length = 4096, # Adjust if using many high-res images
    max_completion_length = 4096, # Space for <think> and <answer>
    log_completions = False,
    beta = 0.0, # KL penalty (start low, 0 with GSPO)
    epsilon = 0.2,
    epsilon_high = 0.28, # one sided
    delta = 1.35, # two sided

    # loss_type = 'bnpo',
    # loss_type = 'grpo',
    # loss_type = 'dr_grpo',
    loss_type = 'dapo',
    importance_sampling_level = "sequence", # Uncomment to enable GSPO

    # --- Hardware Optimization (RTX 5090) ---
    per_device_train_batch_size = 4, # Small because num_generations multiplies this
    gradient_accumulation_steps = 1,
    unsloth_logit_chunk_multiplier = 1, # Higher (up to 8) = less VRAM
    group_by_length = False,
    # warmup_steps = 64,
    warmup_ratio = 0.05,
    weight_decay = 0.01,
    num_train_epochs = 3, # GRPO steps are slow; start with 1 epoch
    max_steps = 256,
    learning_rate = 1e-5, # Lower is safer for reasoning stability
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(), # True with 5090
    logging_steps = 1,
    # use_liger_loss = True, # Drastically cuts memory peaks
    optim = "adamw_8bit", # If using 'paged_adamw_8bit', we can usually push batch size or generations
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    lr_scheduler_type = "cosine",
    seed = seed,
    max_grad_norm = 0.5, # max_grad_norm = 0.1 is quite conservative (usually it's 1.0). It will keep the training very stable, but if you find the model isn't learning fast enough, don't be afraid to bump it to 0.5
    temperature = 1,
    top_p = 0.95,
    min_p = 0.05, # Filters out low-prob "gibberish" tokens

    # --- Windows & Vision Specifics ---
    remove_unused_columns = False, # Essential to keep 'images'

    ddp_find_unused_parameters = False,
    use_vllm = False, # MUST be False on Windows for now to avoid the device_map clash
    dataloader_num_workers = 0, # Force single-process data loading (Critical for Windows)
    report_to = "none",
    mask_truncated_completions = True,
    sync_ref_model = True, # False to save some VRAM

    gradient_checkpointing = True,
    # This is the "magic" line for high-memory 27B models:
    gradient_checkpointing_kwargs = {"use_reentrant": False}
)

# 6. Data Collator (Le chef d'orchestre Texte + Images)
from unsloth import UnslothVisionDataCollator
data_collator = UnslothVisionDataCollator(model, tokenizer)

print(dataset.column_names)
print(f"DEBUG: Dataset length is {len(dataset)}")
print(f"DEBUG: First row keys are {dataset[0].keys()}")
if len(dataset) < 10:
    print("WARNING: Dataset is too small. Check your loading logic!")

def numeric_match_reward(completions, solution, **kwargs):
    # print("""Checks if the specific pricing constants from the image appear in the answer.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res, sol in zip(responses, solution):
        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        # Extract all floats/numbers from the ground truth and the model's answer
        sol_numbers = set(re.findall(r"\d+\.\d+|\d+", sol))
        res_numbers = set(re.findall(r"\d+\.\d+|\d+", match.group(1)))

        if not sol_numbers:
            rewards.append(0.0)
            continue

        # Calculate how many of the required table values were captured
        matches = len(sol_numbers.intersection(res_numbers))
        rewards.append((matches / len(sol_numbers)) * 0.4) # Weighted at 0.4
    return rewards

def length_reward(completions, **kwargs):
    # print("""Penalizes the model for being too wordy.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for res in responses:
        if len(res) > 1200: # If the model 'yaps' over 1200 chars, give a small negative nudge
            rewards.append(-0.1)
        elif len(res) > 1600: # If it's long (>1600), hit it hard
            rewards.append(-0.25)
        elif len(res) > 2400: # If it's extremely long (>2400), hit it harder
            rewards.append(-0.5)
        elif len(res) > 3200: # If it's overly long (>3200), hit it more harder
            rewards.append(-0.75)
        elif len(res) > 4000: # If it's near context limit (>4000), hardest penalty
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

def logic_skeleton_reward(completions, solution, **kwargs):
    # print("""Checks if the formula's numeric constants and logic flow match.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res, sol in zip(responses, solution):
        # Extract content between <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        res_answer = match.group(1)

        # Normalize: remove all whitespace and newlines
        res_norm = re.sub(r"\s+", "", res_answer)
        sol_norm = re.sub(r"\s+", "", sol)

        if res_norm == sol_norm:
            rewards.append(1.0) # Perfect match
        else:
            # Partial credit: Check if key numbers from the image/prompt are present
            # For the UE table: 6.07, 11.51, 32.78, 189.22
            numbers_in_sol = re.findall(r"\d+\.\d+", sol)
            numbers_in_res = re.findall(r"\d+\.\d+", res_norm)

            if not numbers_in_sol:
                rewards.append(0.0)
                continue

            matches = sum(1 for n in numbers_in_sol if n in numbers_in_res)
            score = (matches / len(numbers_in_sol)) * 0.5 # Max 0.5 for partial numbers
            rewards.append(score)

    return rewards

def syntax_and_variable_reward(completions, solution, **kwargs):
    # print("""Ensures variable names are correct and the formula is syntactically valid.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res, sol in zip(responses, solution):
        score = 0.0

        if any(op in res for op in ["if", "{", "}", "(", ")"]):
            score += 0.05 # Tiny nudge to encourage formula-like text

        # 1. Check for XML tags
        if re.search(r"<think>.*?</think>", res, re.DOTALL):
            score += 0.1

        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)

        if match is not None:
            score += 0.1
            formula = match.group(1)

            if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", res, re.DOTALL): # if <think> tag is before <answer> tag
                score += 0.1

            # 2. Variable Check: Do variables in {brackets} match the solution?
            sol_vars = set(re.findall(r"\{(\w+)\}", sol))
            res_vars = set(re.findall(r"\{(\w+)\}", formula))
            if sol_vars and res_vars:
                score += (len(sol_vars.intersection(res_vars)) / len(sol_vars)) * 0.25

            # 3. Syntax Check: Balanced parentheses
            if formula.count('(') == formula.count(')') and '(' in formula:
                score += 0.1

        rewards.append(score)
    return rewards

def valid_arguments_reward(completions, solution, args, **kwargs):
    # print("""Rewards the model for only using variables defined in the schema.""")
    responses = [c[0]["content"] for c in completions]

    rewards = []
    for res, allowed_args, sol in zip(responses, args, solution):
        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        formula = match.group(1)
        # Extract everything inside { }
        used_vars = set(re.findall(r"\{(\w+)\}", formula))
        sol_vars = set(re.findall(r"\{(\w+)\}", sol))

        if not used_vars:
            rewards.append(0.0)
            continue

        # Check how many used variables are actually in the documentation
        allowed_set = set(allowed_args) # Fixes unhashable type list
        valid_vars = used_vars.intersection(allowed_set) # allowed used vars
        correct_vars = used_vars.intersection(sol_vars) # used vars that are in the solution
        hallucinated_vars = used_vars - allowed_set

        # Calculate score: % of variables that are valid
        score = 0.2 * len(valid_vars) / len(used_vars)
        score += 0.8 * len(correct_vars) / len(used_vars)

        # Apply a harsh penalty if even one variable is made up
        if hallucinated_vars:
            score *= 0.5

        rewards.append(score * 0.25) # Weighted lower than logic correctness

    return rewards

def formula_symbols_reward(completions, solution, funcs, ops, **kwargs):
    # print("""Checks if functions and operators match the technical schema.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res, sol, allowed_funcs_list, allowed_ops_list in zip(responses, solution, funcs, ops):
    # valid_funcs and valid_ops come from your dataset columns
        allowed_funcs = set(allowed_funcs_list) # Fixes unhashable type list
        allowed_ops = set(allowed_ops_list)

        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0); continue

        formula = match.group(1)
        score = 0.0

        # 2. Validate Functions (e.g., ceil, if, min)
        found_funcs = set(re.findall(r"(\w+)\s*\(", formula))
        if found_funcs:
            # Reward for using only functions that exist in the documentation
            func_accuracy = len(found_funcs.intersection(allowed_funcs)) / len(found_funcs)
            score += func_accuracy * 0.2

        # 3. Validate Operators (using your 'ops' variable)
        # We check which allowed operators are actually present in the model's answer
        res_ops = set()
        for op in allowed_ops:
             if op in formula:
                   res_ops.add(op)

        # We check which allowed operators are actually present in the ground truth
        sol_ops = set()
        for op in allowed_ops:
             if op in sol:
                   sol_ops.add(op)

        if sol_ops:
            # Reward the model for using the SAME operators required by the solution
            op_match = len(res_ops.intersection(sol_ops)) / len(sol_ops)
            score += op_match * 0.1

            # Penalty check: Did it use operators NOT in the documentation?
            # (This prevents it from hallucinating things like 'AND' or 'EQUALS')
            hallucinated_ops = [op for op in re.findall(r"[^\w\s\(\)\{\}\.,]+", formula)
                                if op not in allowed_ops and len(op) > 0]
            if hallucinated_ops:
                score -= 0.1

        rewards.append(max(0, score))
    return rewards

def check_if_nesting_reward(completions, **kwargs):
    # print("""Rewards proper argument structure in nested if(condition, true, false) blocks.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res in responses:
        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0); continue

        formula = match.group(1)
        if "if(" not in formula:
            rewards.append(0.1) # Neutral reward for simple formulas
            continue

        ifs = formula.count("if(")
        commas = formula.count(",")

        # Standard if requires 2 commas per if: if(a, b, c)
        # We reward closeness to the 2:1 ratio
        target_commas = ifs * 2
        if commas == 0:
            rewards.append(0.0)
        else:
            # Score is 1.0 if ratio is perfect, decaying as it diverges
            ratio_score = 1.0 - abs(commas - target_commas) / max(target_commas, commas)
            rewards.append(max(0.0, ratio_score * 0.2)) # Max 0.2 weighting

    return rewards

def mtow_in_kg_reward(completions, **kwargs):
    # print("""Rewards proper argument structure in nested if(condition, true, false) blocks.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res in responses:
        match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        formula = re.sub(r"[\s\n]*", "", match.group(1))

        # limit_weights = re.findall(r"\d+(?=(==|<=|>=|>|<){MTOW})|(?<={MTOW}(==|<=|>=|>|<))\d+", formula)
        limit_weights = [float(m[0] or m[1]) for m in re.findall(r"\d+({MTOW}\s*(?:==|<=|>=|>|<)\s*(\d+))|((\d+)\s*(?:==|<=|>=|>|<)\s*{MTOW})\d+", formula)]
        if not limit_weights:
            rewards.append(0.0)
            continue

        rewards.append(0.1 * sum(i >= 1000 for i in limit_weights) / len(limit_weights))

    return rewards

def solution_matching_reward(completions, solution, **kwargs):
    # print("""Rewards completions that are close to solution.""")
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for res, sol in zip(responses, solution):
        res = re.sub(r"[\s\n]*", "", res)
        sol = re.sub(r"[\s\n]*", "", sol)

        if res == sol:
            rewards.append(0.2)
        else:
            ratio = SequenceMatcher(None, res, sol).ratio()
            if ratio >= 0.8:
                rewards.append(0.2)
            elif ratio >= 0.5:
                rewards.append(0.4 * (ratio - 0.3))
            else:
                rewards.append(0.0)

    return rewards

def terminal_logging_reward(completions, **kwargs):
    # print("""Directly prints the model's reasoning and answer to the console.""")
    # We only print the first completion of the group to avoid terminal spam
    # completions[0] is the first group, [0]["content"] is the first text
    sample_output = completions[0][0]["content"]

    print("\n" + "="*16)
    print(f"--- MODEL EVOLUTION LOG ---")
    print(f"LATEST OUTPUT SAMPLE:\n{sample_output}")
    # print(f"LATEST OUTPUT SAMPLE:\n{sample_output[:250]}\n...\n{sample_output[-250:]}")
    print("="*16 + "\n")

    return [1.0] * len(completions) # Return a neutral reward

# # Force the trainer to recognize the vision capabilities
# model.is_vision_model = True
# model.is_vision_language_model = True
# model._is_vision_2_seq = True

# model.config.use_cache = False # Required for gradient checkpointing

# model.is_peft_model = True

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 7. Lancement du Trainer
trainer = GRPOTrainer(
    model = model,
    ref_model = None, # CRITICAL: Setting this to None tells TRL to use the base model
    processing_class = tokenizer,
    data_collator = data_collator,
    train_dataset = dataset, # Must have 'prompt' and 'solution' columns
    args = training_args,
    reward_funcs = [
        numeric_match_reward,
        length_reward,
        logic_skeleton_reward,
        syntax_and_variable_reward,
        valid_arguments_reward,
        formula_symbols_reward,
        mtow_in_kg_reward,
        terminal_logging_reward,
        solution_matching_reward
    ]
)

print("--- Starting training ---")
trainer_stats = trainer.train()

# 8. Sauvegarde du modèle
model.save_pretrained("qwen25_pricing_lora_model")
tokenizer.save_pretrained("qwen25_pricing_lora_model")
#model.save_pretrained_gguf("qwen25_pricing_lora_model", tokenizer, quantization_method = ["q8_0"])
model.save_pretrained_merged("qwen25_pricing_lora_model", tokenizer, save_method = "merged_16bit")
print("--- Training finished and saved in 'lora_model' ---")
