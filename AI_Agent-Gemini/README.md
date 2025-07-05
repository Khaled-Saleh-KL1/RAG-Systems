## Gemini AI Agent Chatbot
A desktop chatbot powered by Google's Gemini 2.5 model. This application demonstrates the power of a tool-using AI agent by performing real-time web searches and fetching current weather data to answer user queries.

## Key Features
- **Conversational AI**: Utilizes Google's powerful `gemini-2.5-flash` model for natural and intelligent conversation.
- **AI Agent with Tool Use**: The chatbot autonomously decides when to use tools to answer questions beyond its base knowledge.
- **Real-Time Web Search**: Integrates with the Google Programmable Search Engine API to access up-to-the-minute information from the web.
- **Live Weather Data**: Fetches current weather conditions for any location using the WeatherAPI.
- **Graphical User Interface**: Built with Python's standard Tkinter library for a simple and intuitive user experience.
- **Retrieval-Augmented Generation (RAG)**: Implements the RAG pattern by retrieving external data (from web search or weather APIs) to augment the AI's generation process, leading to more accurate and timely responses.

## Architecture
This application operates in a simple but powerful loop that demonstrates a modern AI agentic workflow:
1. **User Interface (Tkinter)**: The user enters a message in the GUI.
2. **LLM Core (Gemini)**: The message is sent to the Gemini model.
3. **Decision Engine (Function Calling)**: Gemini analyzes the prompt and determines if it can answer from its own knowledge or if it needs to use a tool.
4. **Tool Execution**: If a tool is needed, the model outputs a request to call either the `google_search` or `get_weather` function with the appropriate parameters (e.g., a search query or a city name).
5. **Data Retrieval**: The Python code executes the requested function, calling the external Google Search or WeatherAPI.
6. **Response Generation**: The data retrieved from the API is sent back to the Gemini model, which then generates a final, human-readable response based on this new information.
7. **Display**: The final response is displayed in the Tkinter chat window.

## Getting Started
Follow these instructions to get the project running on your local machine.
Prerequisites
- Python 3.8+
- A Google Cloud account
- A WeatherAPI.com account

**Installation and Setup**
1. Clone the repository:
```
git clone https://github.com/Khaled-Saleh-KL1/AI_Agent-Gemini.git
cd AI_Agent-Gemini
```

2. Create a `requirements.txt` file with the following content
```
google-generativeai
requests
```

3. Install the required packages:
```
pip install -r requirements.txt
```

**Configuration: API Keys**
This project requires three API keys to function correctly. Open the main Python script and add your keys to the configuration section at the top of the file.
```python
# --- Paste your API keys here ---
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"
WEATHER_API_KEY = "PASTE_YOUR_WEATHER_API_KEY_HERE"
GOOGLE_API_KEY = "PASTE_YOUR_GOOGLE_API_KEY_HERE"
SEARCH_ENGINE_ID = "PASTE_YOUR_SEARCH_ENGINE_ID_HERE"
# ------------------------------------
```
- `GEMINI_API_KEY`: Get this from [Google AI Studio](https://aistudio.google.com/app/apikey?authuser=1)
- `WEATHER_API_KEY`: Get this from the [WeatherAPI](https://www.weatherapi.com/) dashboard after signing up for a free account.
- `GOOGLE_API_KEY` and `SEARCH_ENGINE_ID`: These are required for the web search tool. Follow the instructions in the [Google Cloud documentation for the Programmable Search Engine](https://developers.google.com/custom-search/v1/overview?authuser=1) to get both.

## How to Run
Once the dependencies are installed and the API keys are configured, run the application from your terminal:
```
python your_script_name.py
```

## Core Concepts Demonstrated
This project is an excellent educational tool for understanding key concepts in modern AI development:
- **AI Agents**: It showcases a basic but complete agent that can perceive its environment (user queries) and act upon it using tools.
- **Function Calling**: The core mechanism that allows the LLM to request that your code run specific functions to interact with the outside world.
- **RAG**: By using web search, this project implements a real-time RAG system that is not limited to a static set of documents, ensuring its knowledge is always current.

## Author
Khaled Saleh

Note: This is my first project implementing a RAG system, and I am actively learning. If you have any suggestions, find a bug, or want to provide feedback, please feel free to reach out!
- **Email**: khaledsalehkl1@gmail.com
