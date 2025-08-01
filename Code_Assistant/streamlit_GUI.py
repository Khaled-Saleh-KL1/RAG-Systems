import streamlit as st
from workflow import create_workflow, classify_intent_simple
import torch

# Page config
st.set_page_config(
    page_title="AI Workflow Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to fix chat input sizing
st.markdown("""
<style>
    .stChatInput > div > div > textarea {
        min-height: 2.5rem !important;
        max-height: 10rem !important;
        resize: vertical !important;
    }
    
    /* Reset chat input after submission */
    .stChatInput textarea:not(:focus):not(:active) {
        height: 2.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize workflow
@st.cache_resource
def load_workflow():
    return create_workflow()

def main():
    st.title("🤖 AI Workflow: Smart Router → Phi3 → Gemini")
    
    # GPU status
    if torch.cuda.is_available():
        st.success(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ GPU not available")
    
    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Display AI response with sections
                if "intent" in message:
                    with st.expander("🧠 Intent Classification"):
                        st.json(message["intent"])
                
                if "phi3_response" in message:
                    with st.expander("🤖 Phi3 Response"):
                        st.write(message["phi3_response"])
                
                st.subheader("✨ Final Response")
                st.write(message["content"])
    
    # Add a key to force re-render and reset the input size
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    # Chat input with dynamic key to reset size after submission
    user_prompt = st.chat_input(
        "💭 Ask me anything...", 
        key=f"chat_input_{st.session_state.input_key}"
    )
    
    if user_prompt:
        # Increment key to force input reset on next render
        st.session_state.input_key += 1
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    workflow = load_workflow()
                    result = workflow.invoke({
                        "user_prompt": user_prompt,
                        "intent_classification": {},
                        "phi3_response": "",
                        "gemini_response": ""
                    })
                    
                    # Display results in expandable sections
                    with st.expander("🧠 Intent Classification"):
                        intent = result['intent_classification']
                        st.json(intent)
                    
                    with st.expander("🤖 Phi3 Response"):
                        st.write(result['phi3_response'])
                    
                    st.subheader("✨ Final Response")
                    st.write(result['gemini_response'])
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['gemini_response'],
                        "intent": result['intent_classification'],
                        "phi3_response": result['phi3_response']
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # Force input reset with JavaScript
        st.markdown("""
        <script>
        setTimeout(function() {
            const textareas = document.querySelectorAll('.stChatInput textarea');
            textareas.forEach(function(textarea) {
                textarea.style.height = '2.5rem';
                textarea.rows = 1;
            });
        }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        # Force a rerun to refresh the input
        st.rerun()

if __name__ == "__main__":
    main()