"""Simple configuration for AI workflow"""
import os

# Configuration constants
PHI3_MODEL_PATH = "../Week4/model/Phi-3-mini-4k-instruct-q4.gguf"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

def get_gemini_api_key():
    """Get Gemini API key from environment variable"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return api_key
    
    @classmethod
    def validate_config(cls, config: WorkflowConfig) -> bool:
        """Validate the configuration"""
        required_fields = ["phi3_model_path", "gemini_api_key", "gemini_model_name"]
        
        for field in required_fields:
            if not config.get(field):
                print(f"❌ Missing required configuration: {field}")
                return False
        
        return True
