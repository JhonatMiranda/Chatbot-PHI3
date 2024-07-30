# Chatbot-PHI3
A chatbot using Phi3 vision model and Gradio UI

# Table of Contents
1. [Installation](#installation)
2. [Test-the-base-model](#Test-the-base-model)
3. [Running-with-ui](#running-with-ui)

# Installation
First, we need to recreate a conda enviroment to run the codes: 
```bash
conda env create -f environment.yml
# if it's not setted, copy your HF token, found here: https://huggingface.co/settings/account
#to set on Ubuntu:
huggingface-cli login
#paste the copied token
```



# Test-the-base-model
First, we can test the phi3 model using the `sanity_test.py` file. Make sure to stay in the conda enviroment to run this code:

```bash
conda activate phi3

python3 sanity_test.py
```

Then, execute the scripts `scripts\open_llama_sanity_test1.py` and `scripts\open_llama_sanity_test2.py`.

# Running-with-ui

To run the phi3 vision model using Gradio UI, before activate the conda enviroment, execute the `gradio_test.py` file and copy the public URL on the output and paste in your browser.


