import gradio as gr
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# OpenRouter API endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Headers for OpenRouter API
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:7860", # Replace with your site URL in production
    "X-Title": "Gemma Models Demo" # Adding X-Title header for OpenRouter rankings
}

# Available Gemma models on OpenRouter (free tier), ordered from smallest to largest
MODELS = {
    "Google: Gemma 3 1B": "google/gemma-3-1b-it:free",
    "Google: Gemma 3 4B": "google/gemma-3-4b-it:free",
    "Google: Gemma 2 9B": "google/gemma-2-9b-it:free",
    "Google: Gemma 3 12B": "google/gemma-3-12b-it:free",
    "Google: Gemma 3 27B": "google/gemma-3-27b-it:free"
}

# Example prompt templates
PROMPT_TEMPLATES = {
    "Basic": "{query}",
    "Role-based": "You are an expert in {topic}. {query}",
    "Step-by-Step": "Think step by step to solve this problem: {query}",
    "Chain of Thought": "Let's think through this carefully, reasoning one step at a time: {query}",
    "Few-Shot Learning": """Here are a few examples:
    
Input: What is 2+2?
Output: The answer is 4.
    
Input: What is the capital of France?
Output: The capital of France is Paris.
    
Now answer this question: {query}"""
}

def generate_response(query, model, prompt_template, topic="", temperature=0.7, max_tokens=500):
    """Generate a response from the selected model with the selected prompt template."""
    if not OPENROUTER_API_KEY:
        return "⚠️ Please set your OPENROUTER_API_KEY environment variable."
    
    if prompt_template == "Role-based" and not topic:
        topic = "artificial intelligence"
    
    # Format the prompt based on the selected template
    formatted_prompt = PROMPT_TEMPLATES[prompt_template].format(query=query, topic=topic)
    
    # Prepare the payload for the API request
    payload = {
        "model": MODELS[model],
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def update_prompt_preview(query, prompt_template, topic=""):
    """Update the prompt preview based on the selected template."""
    if prompt_template == "Role-based" and not topic:
        topic = "artificial intelligence"
    
    return PROMPT_TEMPLATES[prompt_template].format(query=query, topic=topic)

# Create the Gradio interface
with gr.Blocks(title="Gemma Models Demo") as demo:
    gr.Markdown("# Gemma Models Demonstration")
    gr.Markdown("""
    This demo shows how different Google Gemma models and prompt templates can affect outputs.
    Try typing a query and comparing the results across different models and prompt engineering techniques.
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Query",
                placeholder="Enter your question or task here...",
                lines=3
            )
            
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="Select Gemma Model",
                value=list(MODELS.keys())[0]
            )
            
            template_dropdown = gr.Dropdown(
                choices=list(PROMPT_TEMPLATES.keys()),
                label="Prompt Template",
                value="Basic"
            )
            
            topic_input = gr.Textbox(
                label="Topic/Expertise (for Role-based template)",
                placeholder="e.g., mathematics, history, programming",
                visible=False
            )
            
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            
            submit_button = gr.Button("Generate Response")
        
        with gr.Column():
            prompt_preview = gr.Textbox(
                label="Prompt Preview",
                lines=5,
                interactive=False
            )
            
            response_output = gr.Textbox(
                label="Model Response",
                lines=15,
                interactive=False
            )
    
    # Examples section
    gr.Markdown("## Example Queries")
    examples = gr.Examples(
        examples=[
            ["Explain quantum computing to me"],
            ["Write a short story about a robot learning to feel emotions"],
            ["What are the main causes of climate change?"],
            ["How do I optimize a neural network that's overfitting?"],
            ["Compare and contrast renewable energy sources"]
        ],
        inputs=query_input
    )
    
    # Connect components with events
    template_dropdown.change(
        fn=lambda x: gr.update(visible=(x == "Role-based")),
        inputs=template_dropdown,
        outputs=topic_input
    )
    
    # Update prompt preview when inputs change
    for component in [query_input, template_dropdown, topic_input]:
        component.change(
            fn=update_prompt_preview,
            inputs=[query_input, template_dropdown, topic_input],
            outputs=prompt_preview
        )
    
    # Submit button event
    submit_button.click(
        fn=generate_response,
        inputs=[query_input, model_dropdown, template_dropdown, topic_input, temperature_slider],
        outputs=response_output
    )
    
    # Add explanations of prompt engineering techniques
    gr.Markdown("""
    ## Prompt Engineering Techniques Explained
    
    - **Basic**: Just sends your query directly to the model without any special formatting.
    
    - **Role-based**: Assigns an expert role to the model, which can improve responses for specific domains.
    
    - **Step-by-Step**: Explicitly instructs the model to break down its reasoning, which often improves accuracy.
    
    - **Chain of Thought**: Similar to step-by-step but encourages more detailed reasoning.
    
    - **Few-Shot Learning**: Provides examples of the expected format or reasoning, helping the model understand the task.
    
    ## Gemma Models Information
    
    - **Gemma 3 1B**: The smallest Gemma model (1 billion parameters) with 32,000 token context length
    - **Gemma 3 4B**: A 4 billion parameter model with 131,072 token context length
    - **Gemma 2 9B**: A 9 billion parameter model with 8,192 token context length
    - **Gemma 3 12B**: A 12 billion parameter model with 131,072 token context length
    - **Gemma 3 27B**: The largest Gemma model (27 billion parameters) with 96,000 token context length
    """)

# Instructions for setting up the API key
setup_instructions = """
## Setup Instructions

To use this demo, you need an OpenRouter API key:

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Create a file named `.env` in the same directory as this script with:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Restart this application

"""

if not OPENROUTER_API_KEY:
    demo = gr.Blocks().queue()
    with demo:
        gr.Markdown("# Gemma Models Demo - Setup Required")
        gr.Markdown(setup_instructions)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)