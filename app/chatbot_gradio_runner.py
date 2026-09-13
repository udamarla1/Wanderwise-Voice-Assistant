"""Script to run the gradio as code"""

import gradio as gr
from genai_voice.bots.chatbot import ChatBot
from genai_voice.logger.log_utils import log, LogLevels


# poetry run RunWanderwiseAssistant
def run():
    """Run the Wanderwise Voice Assistant app."""
    chatbot = ChatBot(enable_speakers=True, threaded=True)
    history = []

    # Updated to use Blocks for better control flow (Audio -> Text -> Edit -> Submit)
    def transcribe(audio):
        """Transcribe audio to text"""
        if not audio:
            return ""
        prompt = chatbot.get_prompt_from_gradio_audio(audio)
        log(f"Transcribed prompt: {prompt}", log_level=LogLevels.ON)
        return prompt

    def get_response(text_input):
        """Get Response from Text Input"""
        if not text_input or not text_input.strip():
            return "Please enter some text."
            
        log(f"Processing prompt: {text_input}", log_level=LogLevels.ON)
        response = chatbot.respond(text_input, history)
        history.append([text_input, response])
        return response

    with gr.Blocks(title="Wanderwise Voice Assistant") as demo:
        gr.Markdown("# Wanderwise Voice Assistant")
        
        with gr.Row():
            # Audio Input
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
            
            # Text Input (Editable)
            text_input = gr.Textbox(label="Your Request (Edit if needed)", placeholder="Type or record audio...", lines=3)
            
        submit_btn = gr.Button("Submit", variant="primary")
        output = gr.Textbox(label="Response", interactive=False)
        
        # Events
        # 1. When recording stops, transcribe and fill the text box
        audio_input.stop_recording(fn=transcribe, inputs=audio_input, outputs=text_input)
        
        # 2. When submit is clicked, send text to bot
        submit_btn.click(fn=get_response, inputs=text_input, outputs=output)
        
        # 3. Allow pressing Enter on text box to submit
        text_input.submit(fn=get_response, inputs=text_input, outputs=output)

    demo.launch()


# poetry run RunWanderwiseAudioFromFile
def run_with_file_support():
    """Run the Wanderwise Voice Assistant with file input."""
    chatbot = ChatBot(enable_speakers=True, threaded=True)
    history = []

    def get_response_from_file(file):
        prompt = chatbot.get_prompt_from_file(file)
        response = chatbot.respond(prompt, history)
        history.append([prompt, response])
        return response

    # Approach that doesn't have the warning but uses temp files
    demo = gr.Interface(
        get_response_from_file,
        gr.Audio(sources="microphone", type="filepath"),
        "text",
    )
    demo.launch()
