'''
Base code: temp_test.py
Changes: Using a model from the Gemini family to synthesize the final answer. The tool calls are still generated using Llama 3.1.
'''

import json
import os
import time
os.environ["HF_HOME"] = "../hf_cache"
import re
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# --- New Import for Gemini ---
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai

os.environ["HF_HOME"] = "../hf_cache"

with open("data/llama_prompt.txt", "r") as f:
    context_template = f.read()

with open("data/gemini_prompt_eg_1.txt", "r") as f:
    answer_template = f.read()


# --- Helper Function: Robust JSON Parsing ---
def extract_first_json_array(text: str) -> list:
    """Scans text to find and parse the first valid JSON array."""
    start_idx = text.find('[')
    if start_idx == -1:
        raise ValueError("No starting bracket '[' found in the LLM response.")
        
    open_brackets = 0
    for i in range(start_idx, len(text)):
        if text[i] == '[':
            open_brackets += 1
        elif text[i] == ']':
            open_brackets -= 1
            if open_brackets == 0:
                # We've found the matching closing bracket for the first array
                json_str = text[start_idx:i+1]
                return json.loads(json_str)
                
    raise ValueError("Could not find a matching closing bracket ']'.")


# --- 1. LLM Interface ---
def call_llama_model(prompt: str, tokenizer, model, max_new_tokens) -> str:
    print(f"\n[SYSTEM] Sending prompt to Llama 3.1...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# def call_gemini_model(prompt: str, model, max_new_tokens) -> str:
#     """Calls the Gemini API to synthesize the final answer."""
#     print(f"\n[SYSTEM] Sending prompt to Gemini model...")
#     try:
#         # Note: The tokenizer argument is unused here as Gemini handles tokenization API-side
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 max_output_tokens=max_new_tokens,
#                 temperature=0.1, # Keeping temperature consistent with Llama
#             )
#         )
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error calling Gemini API: {e}")
#         return f"Error: Failed to generate response from Gemini. {str(e)}"

def call_gemini_model(prompt: str, model, max_new_tokens: int, max_retries: int = 5) -> str:
    """Calls the Gemini API to synthesize the final answer with exponential backoff."""
    print(f"\n[SYSTEM] Sending prompt to Gemini model...")
    base_delay = 2  # Starting delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.1, 
                )
            )
            return response.text.strip()
            
        except google_exceptions.ResourceExhausted as e:
            # Catch 429 Quota/Rate Limit errors
            wait_time = base_delay * (2 ** attempt)
            print(f"[WARNING] Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            
        except google_exceptions.ServiceUnavailable as e:
            # Catch 503 Server overloaded errors
            wait_time = base_delay * (2 ** attempt)
            print(f"[WARNING] Service unavailable. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            
        except Exception as e:
            # For other errors, print them. If it's the last attempt, return the error string.
            print(f"Error calling Gemini API: {e}")
            if attempt == max_retries - 1:
                return f"Error: Failed to generate response from Gemini after {max_retries} attempts. {str(e)}"
            
            # Still apply backoff for generic exceptions just in case it's a transient network issue
            wait_time = base_delay * (2 ** attempt)
            time.sleep(wait_time)
            
    return "Error: Max retries exceeded."


# INFO_FILE = "/path/to/your/shared/folder/server_info.txt"

def wait_for_server(info_file):
    print("Waiting for server address file to be created...")
    
    # 1. Wait for the bash script to write the file
    while not os.path.exists(info_file):
        time.sleep(5)
        
    with open(info_file, "r") as f:
        base_url = f.read().strip()
        
    print(f"Found server at {base_url}. Waiting for it to boot...")
    
    # 2. Ping the health endpoint until the model is loaded and ready
    health_url = f"{base_url}/health"
    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print("Server is healthy and ready!")
                return base_url
        except requests.exceptions.ConnectionError:
            # Server hasn't bound to the port yet
            pass
            
        time.sleep(5)

# --- 2. Tool Execution Logic ---
# def execute_tool(tool_name: str, query: str, datasheet_id: str) -> tuple[bool, str]:
def execute_tool(tool_url: str|None, query: str, datasheet_id: str) -> tuple[bool, str]:
    """
    Returns a tuple of (success_boolean, result_string).
    If success is False, the result_string contains the detailed error message.
    """
    # if tool_name not in TOOL_PORTS:
    #     return False, f"Error: Invalid tool {tool_name}"
    
    # port = TOOL_PORTS[tool_name]
    # url = f"http://127.0.0.1:{port}/retrieve/"

    if tool_url is None:
        return False, "Error: Tool URL is None. The server may not be ready."
    
    url = f"{tool_url}/retrieve/"
    
    payload = {
        "query": query,
        "datasheet_id": datasheet_id,
        "top_k": 3
    }
    # print(f"payload for {tool_name}: {payload}")
    
    try:
        # print(f"  -> Executing {tool_name} on port {port} with query: '{query}'...")
        response = requests.post(url, json=payload)
        # print(f"  -> Received response with status code: {response.status_code}")
        
        # Capture detailed server error message on failure
        if response.status_code != 200:
            error_msg = f"Server Error Message: {response.text}"
            # print(f"  -> {error_msg}")
            return False, error_msg
        
        data = response.json()
        return True, str(data)
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Error during retrieval: {str(e)}"
        # print(f"  -> {error_msg}")
        return False, error_msg


# --- 3. Main Agent Loop ---
def process_all_val_cases():
    data_dir = "datasheet-qa-challenge/dataset"
    val_filename = "test.jsonl"
    VAL_DATASET_PATH = os.path.join(data_dir, val_filename)
    # VAL_DATASET_PATH = "datasheet-qa-challenge/dataset/val.jsonl"
    # RESULT_DATASET_PATH = f"data/result_{val_filename}"
    RESULT_DATASET_PATH = f"data/trial_gemini_test.jsonl"

    # 1. Load Base Model (Llama 3.1)
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter_path = None # "logs/sft_1"
    print(f"adapter_path: {adapter_path}")

    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if adapter_path is not None and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Initialize Gemini model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
    
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as the default for standard text synthesis tasks
    gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite') 
    # gemini_tokenizer = None  # The Gemini SDK does not require passing a local tokenizer
    print(f"Initialized Gemini model: {gemini_model.model_name}")

    # Initialize connection
    text_info_file = "logs/text_server_info.txt"
    text_url = wait_for_server(text_info_file)

    table_info_file = "logs/table_server_info.txt"
    table_url = wait_for_server(table_info_file)

    tool_urls = {
        "text_search": text_url,
        "table_search": table_url,
        "figure_search": None  # Assuming figure_search server is not yet implemented
    }

    max_new_tokens = 200
    # If RESULT_DATASET_PATH does not exist, create it and write the header
    if not os.path.exists(RESULT_DATASET_PATH):
        with open(RESULT_DATASET_PATH, "w") as f:
            f.write("")  # Create an empty file
    
    # If RESULT_DATASET_PATH exists, read existing datasheet_ids to avoid duplicates
    # existing_datasheet_ids = set()
    # if os.path.exists(RESULT_DATASET_PATH):
    #     with open(RESULT_DATASET_PATH, "r") as f:
    #         for line in f:
    #             if line.strip():
    #                 try:
    #                     record = json.loads(line)
    #                     existing_datasheet_ids.add(record.get("datasheet_id", ""))
    #                 except json.JSONDecodeError:
    #                     print(f"Warning: Skipping invalid JSON line in {RESULT_DATASET_PATH}: {line.strip()}")

    # 2. Iterate through all validation examples
    with open(VAL_DATASET_PATH, "r") as f_in, open(RESULT_DATASET_PATH, "w") as f_out:
        i = 0
        for line in f_in:
            if not line.strip():
                continue
                
            val_example = json.loads(line)
            question = val_example.get("question", "")
            datasheet_id = val_example.get("datasheet_id", "")
            
            print("="*50)
            print(f"Datasheet ID: {datasheet_id}")
            print(f"Question: {question}")
            print("="*50)

            tool_prompt = f"""{context_template}\nquestion: {question}\ndatasheet_id: {datasheet_id}"""

            # llm_tool_response = call_llama_model(tool_prompt, tokenizer, model, max_new_tokens=max_new_tokens) 
            # print(f"llm_tool_response: {llm_tool_response}")
            llm_tool_response = call_gemini_model(tool_prompt, gemini_model, max_new_tokens=max_new_tokens)
            
            predicted_answer = ""
            generated_tool_calls = []
            
            # --- Parsing tool calls ---
            try:
                generated_tool_calls = extract_first_json_array(llm_tool_response)
                print(f"Successfully extracted JSON: {generated_tool_calls}")
            except Exception as e:
                print(f"Failed to parse LLM tool output as JSON: {e}")
                print(f"LLM Response: {llm_tool_response}")
                print("="*50)
                predicted_answer = f"Error: Failed to parse LLM tool output. {str(e)}"
            
            # Step 2: Execute the tool calls if parsing was successful
            retrieved_contexts = []
            server_error_occurred = False
            
            if not predicted_answer: # Only run tools if parsing succeeded
                for call in generated_tool_calls:
                    tool_name = call.get("tool")
                    query = call.get("query")
                    
                    # success, result = execute_tool(tool_name, query, datasheet_id)
                    success, result = execute_tool(tool_urls.get(tool_name), query, datasheet_id)
                    
                    if not success:
                        server_error_occurred = True
                        predicted_answer = result # Set the final answer to the detailed error message
                        break # Stop executing further tools for this example
                        
                    retrieved_contexts.append(f"Result from {tool_name} for '{query}':\n{result}")

            # Step 3: Prompt LLM to synthesize final answer (if no server errors occurred)
            if not server_error_occurred and not predicted_answer:
                combined_context = "\n\n".join(retrieved_contexts)

                answer_prompt = f"""{answer_template}\nquestion: {question}\ndatasheet_id: {datasheet_id}\ngenerated_tool_calls: {generated_tool_calls}\nRetrieved Context:\n{combined_context}\nFinal Answer:"""

                print(f"answer_prompt: {answer_prompt}")
                # raw_answer = call_llama_model(answer_prompt, tokenizer, model, max_new_tokens=max_new_tokens)
                raw_answer = call_gemini_model(answer_prompt, gemini_model, max_new_tokens=max_new_tokens)
                predicted_answer = raw_answer.strip() if raw_answer else "LLM Answer Simulation Failed"

            # Step 4: Format final output and write to jsonl
            final_output = {
                "datasheet_id": datasheet_id,
                "question": question,
                "tool_calls_generated": generated_tool_calls,
                "predicted_answer": predicted_answer
            }

            # print("\nFINAL AGENT OUTPUT FOR CURRENT RECORD")
            # print(json.dumps(final_output, indent=2))
            
            # Write line to result_val.jsonl
            f_out.write(json.dumps(final_output) + "\n")
            f_out.flush() # Ensure it writes to disk immediately

            i += 1
            if i % 10 == 0:
                print(f"\n[SYSTEM] Processed {i} records so far. Results saved to {RESULT_DATASET_PATH}")
            # if i >= 10:
            #     print(f"\n[SYSTEM] Reached processing limit of 10 records for testing purposes. Stopping early.")
            #     break

            time.sleep(10)

    print(f"\n[SYSTEM] Finished processing. Results saved to {RESULT_DATASET_PATH}")


if __name__ == "__main__":
    process_all_val_cases()