from unsloth import FastVisionModel
from datasets import load_dataset


dataset = load_dataset("MananSuri27/Flowchart2Mermaid", split="validation")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2-VL-7B-Instruct", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
    )
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[70]["image"]
instruction = "You are given a flowchart, with nodes labelled in red with alphabets. Generate mermaid code, that structurally represents the flowcharts, while making references to the red letter labels for the flowchart nodes."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

print(dataset[70]["text"])

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512,
                   use_cache = True, temperature = 1.5, min_p = 0.1)