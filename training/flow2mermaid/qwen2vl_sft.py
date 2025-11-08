from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

train_dataset, eval_dataset = load_dataset("MananSuri27/Flowchart2Mermaid", split = ["train", "validation"])

instruction = "You are given a flowchart, with nodes labelled in red with alphabets. Generate mermaid code, that structurally represents the flowcharts, while making references to the red letter labels for the flowchart nodes."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }


train_converted_dataset = [convert_to_conversation(sample) for sample in train_dataset]
eval_converted_dataset = [convert_to_conversation(sample) for sample in eval_dataset]


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_converted_dataset,
    eval_dataset = eval_converted_dataset,
    args = SFTConfig(
        output_dir="Qwen2-7b-instruct-sft-flowchart",
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 5,
        warmup_steps = 10,
        # max_steps = 30,
        num_train_epochs = 3, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 392005,
        report_to = "wandb",     # For Weights and Biases
        logging_steps=50,  # Steps interval for logging
        eval_steps=50,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)


trainer_stats = trainer.train()

print(trainer_stats)


model.push_to_hub("MananSuri27/Qwen2-7b-instruct-sft-flowchart") 
tokenizer.push_to_hub("MananSuri27/Qwen2-7b-instruct-sft-flowchart") 

model.push_to_hub_merged("MananSuri27/Qwen2-7b-instruct-sft-flowchart_merged", tokenizer)