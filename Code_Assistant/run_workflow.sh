#!/bin/bash

# AI Workflow Runner Script
echo "🚀 AI Workflow Assistant: Smart Router → Phi3 → Gemini"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed!"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY environment variable not set!"
    echo "⚠️ Please run: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Interface selection menu
echo ""
echo "🎯 Choose your interface:"
echo "  [S/s] - Streamlit Interface (Chat-based UI)"
echo "  [G/g] - Gradio Interface (Interactive UI)"
echo "  [Q/q] - Quit"
echo ""

# Function to launch Streamlit
launch_streamlit() {
    echo "▶️ Launching Streamlit interface..."
    echo "🌐 Opening at: http://localhost:8501"
    streamlit run streamlit_GUI.py
}

# Function to launch Gradio
launch_gradio() {
    echo "▶️ Launching Gradio interface..."
    echo "🌐 Opening at: http://localhost:7860"
    python3 Gradio_GUI.py
}

# Main selection loop
while true; do
    read -p "👉 Enter your choice [S/G/Q]: " choice
    
    case $choice in
        [Ss])
            launch_streamlit
            break
            ;;
        [Gg])
            launch_gradio
            break
            ;;
        [Qq])
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter S, G, or Q."
            ;;
    esac
done

echo "✅ Interface session completed!"