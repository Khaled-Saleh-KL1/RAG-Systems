import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
import datetime
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
import requests
import threading

# --- Paste your API keys here ---
GEMINI_API_KEY = ""  # Replace with your Gemini Flash 2.5 API key
WEATHER_API_KEY = "" # Replace with your WeatherAPI.com key
SEARCH_ENGINE_ID = "" # Replace with your Google Custom Search Engine ID
GOOGLE_API_KEY = "" # Replace with your Google API key
# ------------------------------------

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# --- Tool Definitions ---

def google_search(query: str) -> str:
    """
    This tool performs a web search using the Google Custom Search JSON API.
    Args:
        query (str): The search query to use.
    """
    print(f"--- Performing Google search for: {query} ---")
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': GOOGLE_API_KEY,
            'cx': SEARCH_ENGINE_ID,
            'num': 5  # Request top 5 results
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        search_results = response.json()

        # Extract snippets from the results
        items = search_results.get("items", [])
        if not items:
            return "Tool returned no results. Inform the user that you couldn't find anything."

        # Format the results for the language model
        snippets = [f"Title: {item.get('title', '')}\nSnippet: {item.get('snippet', '')}" for item in items]
        return "\n\n".join(snippets)

    except Exception as e:
        print(f"--- Google Search failed! Error: {e} ---")
        return f"Tool Error: The search failed. Do not try again. Reason: {e}"

def get_weather(city: str) -> str:
    """A tool that fetches the current weather for a specified city.
    Args:
        city: Name of the city to get the weather for.
    """
    if not WEATHER_API_KEY:
        return "Error: Weather API key is not configured."

    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'error' in data:
            return f"Could not fetch weather for {city}: {data['error']['message']}"

        location = data["location"]["name"]
        condition = data["current"]["condition"]["text"]
        temp_c = data["current"]["temp_c"]
        
        return f"The current weather in {location} is {condition} with a temperature of {temp_c}°C."
    
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error fetching weather: {http_err}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

class ChatApplication:
    def __init__(self, root):
        self.root = root
        root.title("Conversational AI")

        self.chat_history = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD, width=60, height=20)
        self.chat_history.pack(padx=10, pady=10)

        self.message_entry = Entry(root, width=50)
        self.message_entry.pack(pady=5, padx=10, side=tk.LEFT, expand=True, fill=tk.X)
        self.message_entry.bind("<Return>", self.send_message_event)

        self.send_button = Button(root, text="Send", command=self.send_message_event)
        self.send_button.pack(pady=5, padx=10, side=tk.RIGHT)

        system_prompt = """
        You are a helpful AI assistant, you can use the following tools to answer questions:
        - google_search: Use this tool to search the web for information.
        - get_weather: Use this tool to get the current weather in a given location.
        You will be provided with a question, and you should use the appropriate tools to find the answer.
        Today's date is {current_date}. You MUST use this as the current date for all requests.
        When a user asks about information from the current year, treat it as a request for current events and use your search tool.
        Do not refuse to search for information by claiming it's in the future. Your internal knowledge is outdated, so you must rely on your tools for anything recent.
        """

        model = GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=[google_search, get_weather],
            system_instruction=system_prompt
        )

        self.chat = model.start_chat(enable_automatic_function_calling=True)
        self.add_to_chat_history("AI: Hello! How can I assist you today?")

    def add_to_chat_history(self, message):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.config(state='disabled')
        self.chat_history.yview(tk.END)

    def send_message_event(self, event=None):
        user_input = self.message_entry.get()
        if user_input.strip():
            self.add_to_chat_history(f"You: {user_input}")
            self.message_entry.delete(0, tk.END)
            
            # Disable input while processing
            self.message_entry.config(state='disabled')
            self.send_button.config(state='disabled')

            # Run API call in a separate thread to keep the GUI responsive
            thread = threading.Thread(target=self.get_ai_response, args=(user_input,))
            thread.start()

    def get_ai_response(self, user_input):
        try:
            response = self.chat.send_message(user_input)
            self.root.after(0, self.update_chat_with_ai_response, f"AI: {response.text}")
        except Exception as e:
            self.root.after(0, self.update_chat_with_ai_response, f"AI: An error occurred: {e}")

    def update_chat_with_ai_response(self, response_text):
        self.add_to_chat_history(response_text)
        
        # Re-enable input
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.message_entry.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()