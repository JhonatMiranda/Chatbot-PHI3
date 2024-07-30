from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import re

# Carregar o modelo e o tokenizer
processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-vision-128k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)


def clean_response(response):
    # Define a list of patterns to remove
    patterns_to_remove = [
        r'\[RESPONSE\]', 
        r'\[CONTEXT\]', 
        r'\[QUESTION\]', 
        r'EOS', 
        r'Assistant:',
        r'User:.*'
    ]
    
    # Remove each pattern from the response
    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response)
    
      
    return response

# Definindo a função predict com state e model prediction
def predict(input, history=[]):
       
    instruction = 'Instruction: given a dialog context, you need to response empathically'
    knowledge = '  '
    s = list(sum(history, ()))      
    dialog = ' EOS ' .join(s)
    query = f"[CONTEXT] {dialog} [ENDOFCONTEXT] User: {input} {knowledge}"
    
    messages = [ 
        {"role": "system", "content": f"{instruction}"}, 
        {"role": "user", "content": f"{query}"},  
    ]  
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, return_tensors="pt").to("cuda:0")

    generation_args = {
      "max_new_tokens": 500,
      "do_sample": False,
    }
    
    print("query:\n",query)
    print("-------------------------")

    # Gera a saída usando o modelo
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print("Response:\n",response)
    print("-------------------------")

    response_clean = clean_response(response)

    print("Response Cleaned:\n", response_clean)
    print("-------------------------")
    
    history.append((input, ("Assistant: " + response_clean)))

    return history, history

# Criando a interface do chatbot Gradio
import gradio as gr

gr.Interface(fn=predict,
             inputs=["text", 'state'],
             outputs=["chatbot", 'state']).launch(debug=True, share=True)
