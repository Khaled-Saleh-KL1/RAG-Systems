import gradio as gr
from workflow import create_workflow, classify_intent_simple
import torch
import json

# Initialize workflow
workflow = None

def load_workflow():
    """Load the workflow once and cache it"""
    global workflow
    if workflow is None:
        workflow = create_workflow()
    return workflow

def get_gpu_status():
    """Get GPU status information"""
    if torch.cuda.is_available():
        return f"🔥 GPU: {torch.cuda.get_device_name(0)}"
    else:
        return "⚠️ GPU not available - using CPU"

def process_message(message, history):
    """Process user message through the AI workflow"""
    if not message.strip():
        return history, ""
    
    try:
        # Load workflow
        workflow = load_workflow()
        
        # Process through workflow
        result = workflow.invoke({
            "user_prompt": message,
            "intent_classification": {},
            "phi3_response": "",
            "gemini_response": ""
        })
        
        # Format the response with collapsible sections (similar to Streamlit expandable)
        intent_json = json.dumps(result['intent_classification'], indent=2)
        
        # Create formatted response with collapsible sections using HTML details/summary
        bot_response = f"""✨ **Final Gemini Response:**

{result['gemini_response']}

---

<details>
<summary><b>🧠 Intent Classification</b> (Click to expand)</summary>

```json
{intent_json}
```

</details>

<details>
<summary><b>🤖 Phi3 Response</b> (Click to expand)</summary>

{result['phi3_response']}

</details>"""
        
        # Add to history
        history.append([message, bot_response])
        
        return history, ""
        
    except Exception as e:
        error_response = f"❌ **Error:** {str(e)}"
        history.append([message, error_response])
        return history, ""

def clear_chat():
    """Clear the chat history"""
    return [], ""

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    
    /* Style for collapsible sections */
    details {
        margin: 10px 0;
        padding: 8px 12px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #313131;
    }
    
    details summary {
        cursor: pointer;
        font-weight: bold;
        padding: 8px 0;
        color: #333;
        user-select: none;
    }
    
    details summary:hover {
        color: #313131;
    }
    
    details[open] {
        background-color: #313131;
        border-color: #ccc;
    }
    
    details[open] summary {
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 10px;
        padding-bottom: 8px;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=css,
        title="AI Workflow Assistant"
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 🤖 AI Workflow Assistant: Smart Router → Phi3 → Gemini
            
            An intelligent AI workflow that combines intent classification, Microsoft's Phi3, and Google's Gemini AI.
            """
        )
        
        # GPU Status
        gpu_status = gr.Markdown(get_gpu_status())
        
        # Chat Interface
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="💭 Ask me anything...",
                        label="Your Message",
                        lines=2,
                        max_lines=5,
                        show_label=False,
                        container=False
                    )
                    
                with gr.Row():
                    send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### ℹ️ How it works:
                    
                    1. **🎯 Intent Classification** - Analyzes your prompt
                    2. **🤖 Phi3 Processing** - Local AI processing
                    3. **✨ Gemini Enhancement** - Main response displayed
                    4. **📋 Expandable Details** - Click to view technical details
                    
                    ### 💡 Tips:
                    - **Gemini response** is shown by default
                    - **Click details** to expand intent classification
                    - **Click details** to expand Phi3 response  
                    - Press **Enter** for new lines
                    - Use **Shift+Enter** to send
                    - Click **Clear Chat** to reset
                    
                    ### 🎯 Interface Features:
                    - **Main Response**: Gemini output prominently displayed
                    - **Collapsible Sections**: Technical details hidden by default
                    - **Clean Layout**: Focus on the final answer
                    """
                )
        
        # Event handlers
        def submit_message(message, history):
            return process_message(message, history)
        
        # Button click events
        send_btn.click(
            fn=submit_message,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input],
            show_progress="full"
        )
        
        # Enter key event
        message_input.submit(
            fn=submit_message,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input],
            show_progress="full"
        )
        
        # Clear chat event
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, message_input]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            
            **Author:** Khaled Saleh | **Email:** khaledsalehkl1@gmail.com
            """
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Loading AI Workflow Assistant...")
    print("📡 Initializing models...")
    
    try:
        # Pre-load workflow for faster first response
        load_workflow()
        print("✅ Workflow loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load workflow: {e}")
    
    # Create and launch interface
    interface = create_interface()
    
    print("🌐 Starting Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        inbrowser=True,         # Open in browser automatically
        show_error=True,        # Show detailed errors
        quiet=False             # Show startup logs
    )

if __name__ == "__main__":
    main()
