# 🧠 Prompt Engineering Interactive Lab

An interactive demo that showcases how different prompt engineering techniques affect AI outputs when working with large language models.

## About

This application demonstrates various prompt engineering techniques using Google's Gemma models via OpenRouter API. Users can experiment with the same query across different prompt techniques to understand how the framing of a prompt dramatically impacts the quality and nature of AI responses.

## Features

- Compare multiple prompt engineering techniques side-by-side
- Choose from various Gemma model sizes (1B to 27B parameters)
- Adjust temperature to control creativity vs precision
- Real-time prompt preview
- Detailed explanations of each technique

## Prompt Techniques Included

- **Basic Prompting**: Direct questions without additional context
- **Role-Based Prompting**: Assigning expertise personas to the model
- **Step-by-Step Reasoning**: Requesting methodical thinking
- **Chain of Thought**: Encouraging careful, sequential reasoning
- **Few-Shot Learning**: Demonstrating examples before the actual prompt

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Set up your own OpenRouter API key:
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Create a `.env` file with your API key:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```
   - A default API key is provided but has limited quota

3. Run the application:
   ```
   python prompt_engineering_demo.py
   ```

4. Open your browser at `http://localhost:7860`

## Usage

1. Enter your query in the text box
2. Select a Gemma model size
3. Choose a prompt technique
4. Adjust the temperature setting if desired
5. Click "Generate Response"
6. View the formatted prompt and resulting AI response
7. Try the same query with different techniques to compare outcomes

## Related Resources

This demo is a companion to the blog post ["What is Prompt Engineering?"](https://slyracoon23.github.io/blog/posts/2025-03-15_what_is_prompt_engineering.html)

## License

[Add your license information here] 