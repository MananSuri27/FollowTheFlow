import os
from tqdm import tqdm
from PIL import Image
from unsloth import FastVisionModel
from transformers import AutoTokenizer

# Load the model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="MananSuri27/Qwen2-7b-instruct-sft-flowchart",  # Your pre-trained model
    load_in_4bit=True  # Adjust based on your setup
)
FastVisionModel.for_inference(model)  # Enable for inference!


# Function to generate mermaid code for an image
def generate_mermaid_code(image_path):
    image = Image.open(image_path)  # Open the image
    instruction = "You are given a flowchart, with nodes labelled in red with alphabets. Generate mermaid code, that structurally represents the flowcharts, while making references to the red letter labels for the flowchart nodes. Make sure that node labels are accurate."
    

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    # Tokenize input
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate mermaid code
    generated_code = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )
    
    # Decode the generated output

    prompt_length = inputs['input_ids'].shape[1]

    answer = tokenizer.decode(generated_code[0][prompt_length:], skip_special_tokens=True)
    return answer

# Main function to process a directory
def process_directory(directory_path):
    # Iterate over subdirectories using TQDM
    for subdir in tqdm(os.listdir(directory_path), desc="Processing subdirectories"):
        subdir_path = os.path.join(directory_path, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Define the image path and check if it exists
        image_path = os.path.join(subdir_path, "segmentation.png")
        if os.path.exists(image_path):
            # Generate mermaid code for the image
            mermaid_code = generate_mermaid_code(image_path)
            
            # Save the mermaid code to a text file
            mermaid_file_path = os.path.join(subdir_path, "mermaid.txt")
            with open(mermaid_file_path, "w") as mermaid_file:
                mermaid_file.write(mermaid_code)
        else:
            print(f"Image not found in {subdir_path}")

# Example usage (uncomment to run):
# process_directory("./data/images")
