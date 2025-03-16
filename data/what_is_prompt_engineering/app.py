import gradio as gr
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Headers for OpenRouter API
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:7860", # Replace with your site URL in production
    "X-Title": "Prompt Engineering Lab" # Updated title for OpenRouter rankings
}

# Available Gemma models on OpenRouter (free tier), ordered from smallest to largest
MODELS = {
    "Google: Gemma 3 1B": "google/gemma-3-1b-it:free",
    "Google: Gemma 3 4B": "google/gemma-3-4b-it:free",
    "Google: Gemma 2 9B": "google/gemma-2-9b-it:free",
    "Google: Gemma 3 12B": "google/gemma-3-12b-it:free",
    "Google: Gemma 3 27B": "google/gemma-3-27b-it:free"
}

# Enhanced prompt templates with better descriptions
PROMPT_TEMPLATES = {
    "None": "{query}",
    "Role-based": "You are an expert in {topic}.\n\n{query}",
    "Step-by-Step": "{query}\n\nThink step by step to solve this problem.",
    "Chain of Thought": "{query}\n\nLet's think through this carefully, reasoning one step at a time.",
    "Few-Shot Learning": """Here are a few examples:

Input: What is 2+2?
Output: The answer is 4.

Input: What is the capital of France?
Output: The capital of France is Paris.

Now answer this question: 

{query}"""
}

# Descriptions of each technique for the UI
TECHNIQUE_DESCRIPTIONS = {
    "None": "Sends your query directly to the model without any prompt engineering technique applied. This serves as a baseline to compare other techniques against.",
    
    "Role-based": "Assigns an expert role to the model, which can improve responses for specific domains by framing the model's perspective. This technique leverages the model's training on expert writing styles.",
    
    "Step-by-Step": "Appends an instruction to think step-by-step at the end of your query, which often improves accuracy on complex problems by encouraging methodical thinking.",
    
    "Chain of Thought": "Appends a request to think carefully with step-by-step reasoning at the end of your query. This approach is particularly effective for complex reasoning tasks.",
    
    "Few-Shot Learning": "Provides examples of the expected format or reasoning before your query, helping the model understand the task through demonstration rather than instruction. This technique is powerful when you need specific output formats or reasoning patterns."
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

def update_technique_description(prompt_template):
    """Update the technique description based on the selected template."""
    return TECHNIQUE_DESCRIPTIONS[prompt_template]

# Create the Gradio interface
with gr.Blocks(title="Prompt Engineering Interactive Lab") as demo:
    gr.Markdown("# 🧠 Prompt Engineering Interactive Lab")
    gr.Markdown("""
    This interactive lab demonstrates how different prompt engineering techniques can dramatically affect AI outputs.
    
    Experiment with various techniques and see how the same query produces different results based on how you frame your prompt.
    This is a hands-on companion to the blog post ["What is Prompt Engineering?"](https://slyracoon23.github.io/blog/posts/2025-03-15_what_is_prompt_engineering.html)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Query",
                placeholder="Enter your question or task here...",
                lines=3
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        label="Select Model",
                        value=list(MODELS.keys())[2]  # Default to 9B model for better results
                    )
                
                with gr.Column(scale=1):
                    template_dropdown = gr.Dropdown(
                        choices=list(PROMPT_TEMPLATES.keys()),
                        label="Prompt Technique",
                        value="None"
                    )
            
            topic_input = gr.Textbox(
                label="Topic/Expertise (for Role-based technique)",
                placeholder="e.g., mathematics, history, programming",
                visible=False
            )
            
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature (Creativity vs Precision)",
                info="Lower values = more precise, higher values = more creative"
            )
            
            submit_button = gr.Button("Generate Response", variant="primary")
        
        with gr.Column(scale=1):
            technique_description = gr.Markdown()
            
            prompt_preview = gr.Textbox(
                label="Prompt Preview",
                lines=5,
                interactive=False
            )
            
            response_output = gr.Textbox(
                label="AI Response",
                lines=15,
                interactive=False
            )
    
    # Examples section with real-world prompting scenarios
    gr.Markdown("## Example Prompting Scenarios")
    examples = gr.Examples(
        examples=[
            ["Explain how transformers work in machine learning"],
            ["Compare and contrast renewable energy sources"],
            ["What are three strategies to improve critical thinking?"],
            ["Design a simple algorithm to find duplicate elements in an array"],
            ["What are the ethical implications of AI in healthcare?"]
        ],
        inputs=query_input
    )
    
    # Connect components with events
    template_dropdown.change(
        fn=lambda x: gr.update(visible=(x == "Role-based")),
        inputs=template_dropdown,
        outputs=topic_input
    )
    
    # Update technique description when template changes
    template_dropdown.change(
        fn=update_technique_description,
        inputs=template_dropdown,
        outputs=technique_description
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
    
    # Add explanations of prompt engineering and its impact
    gr.Markdown("""
    ## Understanding Prompt Engineering
    
    Prompt engineering is the practice of crafting inputs to AI systems to elicit desired outputs. It's a key skill for effectively using large language models.
    
    ### Why Prompt Engineering Matters
    
    The same model can produce dramatically different results based solely on how you frame your prompt. This demo lets you experience this firsthand by comparing different techniques:
    
    - **Basic Prompting**: Direct questions yield direct answers, but may lack depth or context
    - **Role-Based Prompting**: Giving the AI a persona or expertise lens changes its perspective
    - **Step-by-Step Reasoning**: Requesting explicit reasoning steps improves accuracy for complex tasks
    - **Chain of Thought**: Extended reasoning that connects concepts leads to more comprehensive answers
    - **Few-Shot Learning**: Showing examples of desired outputs helps the model understand your expectations
    
    ### Experiment Tips
    
    - Try the same query with different techniques to see how responses vary
    - Adjust the temperature to see how it affects output creativity vs. precision
    - For complex questions, compare basic prompting with reasoning-based techniques
    - For domain-specific questions, try role-based prompting with relevant expertise
    
    This demo uses the Google Gemma model family via OpenRouter's API.
    """)
    
    # Instructions for setting up the API key
    gr.Markdown("""
    ## Setup Information
    
    This demo uses the OpenRouter API to access Gemma models. The default API key has limited quota.
    
    For unlimited use:
    1. Sign up at [OpenRouter](https://openrouter.ai/)
    2. Get your API key from the dashboard
    3. Create a `.env` file in this directory with: `OPENROUTER_API_KEY=your_api_key_here`
    """)

# Handle case when API key is not set
if not OPENROUTER_API_KEY:
    demo = gr.Blocks().queue()
    with demo:
        gr.Markdown("# Prompt Engineering Lab - Setup Required")
        gr.Markdown("""
        ## Setup Instructions
        
        To use this demo, you need an OpenRouter API key:
        
        1. Sign up at [OpenRouter](https://openrouter.ai/)
        2. Get your API key from the dashboard
        3. Create a file named `.env` in the same directory as this script with:
           ```
           OPENROUTER_API_KEY=your_api_key_here
           ```
        4. Restart this application
        """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)