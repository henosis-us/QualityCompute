# pricing.py
# Contains model pricing information and helper functions for the backend.

import re
import logging # Using logging instead of print for warnings is better practice in backend

# Configure basic logging
# In a real app, you'd configure logging more robustly (e.g., in Flask app setup)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# For simplicity here, we'll use print for warnings as in the JS version

BASE_PRICING_PER_MILLION_TOKENS = {
    "gemini-1.5-flash-latest":           { "input": 0.075,   "output": 0.300,   "category": "Google" },
    "gemini-2.5-flash":                  { "input": 0.150,   "output": 0.600,   "category": "Google" },
    "gemini-2.0-flash-lite-preview":     { "input": 0.075,   "output": 0.300,   "category": "Google" },
    "gemini-2.5-flash-thinking":         { "input": 0.150,   "output": 3.500,   "category": "Google" },
    "gemini-1.5-flash-8b":               { "input": 0.0375,  "output": 0.150,   "category": "Google" },
    "gemini-2.5-pro-exp-03-25":          { "input": 1.250,   "output": 10.000,  "category": "Google" },

    "gpt-3.5-turbo":                     { "input": 0.500,   "output": 1.500,   "category": "OpenAI" },
    "gpt-4-turbo":                       { "input": 10.000,  "output": 30.000,  "category": "OpenAI" },
    "gpt-4o-2024-11-20":                 { "input": 2.500,   "output": 10.000,  "category": "OpenAI" },
    "chatgpt-4o-latest":                 { "input": 5.000,   "output": 15.000,  "category": "OpenAI" },

    "claude-3-5-haiku-20241022":         { "input": 1.000,   "output": 5.000,   "image_token_ratio": 750, "category": "Anthropic" },
    "claude-3-5-sonnet-20241022":        { "input": 3.000,   "output": 15.000,  "image_token_ratio": 750, "category": "Anthropic" },
    "claude-3-7-sonnet-20250219":        { "input": 3.000,   "output": 15.000,  "image_token_ratio": 750, "category": "Anthropic" },
    "claude-3-7-sonnet-thinking":        { "input": 3.000,   "output": 15.000,  "image_token_ratio": 750, "category": "Anthropic" },
    "claude-3-opus-20240229":            { "input": 15.000,  "output": 75.000,  "image_token_ratio": 750, "category": "Anthropic" },
    "claude-3-haiku-20240307":           { "input": 0.250,   "output": 1.250,   "image_token_ratio": 750, "category": "Anthropic" },

    "deepseek-chat":                     { "input": 0.140,   "output": 0.280,   "category": "DeepSeek" },
    "deepseek-reasoner":                 { "input": 0.550,   "output": 2.190,   "category": "DeepSeek" },
    "DeepSeek-R1-Zero":                  { "input": 0.550,   "output": 2.190,   "category": "DeepSeek" },
    "DeepSeek-R1":                       { "input": 0.550,   "output": 2.190,   "category": "DeepSeek" },

    "o1-mini":                           { "input": 3.000,   "output": 12.000,  "category": "OpenAI" },
    "o3":                                { "input": 10.000,  "output": 40.000,  "category": "OpenAI" },
    "o1-pro":                            { "input": 150.000, "output": 600.000, "category": "OpenAI" },
    "o4-mini":                           { "input": 1.100,   "output": 4.400,   "category": "OpenAI" },

    "gpt-4.5-preview":                   { "input": 75.000,  "output": 150.000, "category": "OpenAI" },

    "gpt-4o-mini":                       { "input": 0.150,   "output": 0.600,   "category": "OpenAI" },
    "gpt-4.1":                           { "input": 2.000,   "output": 8.000,   "category": "OpenAI" },
    "gpt-4.1-mini":                      { "input": 0.400,   "output": 1.600,   "category": "OpenAI" },
    "gpt-4.1-nano":                      { "input": 0.100,   "output": 0.400,   "category": "OpenAI" },

    "grok-3-beta":                       { "input": 3.000,   "output": 15.000,  "category": "X.AI" },
    "grok-3-fast-beta":                  { "input": 5.000,   "output": 25.000,  "category": "X.AI" },
    "grok-3-mini-beta":                  { "input": 0.300,   "output": 0.500,   "category": "X.AI" },
    "grok-3-mini-fast-beta":             { "input": 0.600,   "output": 4.000,   "category": "X.AI" },
}

# Map the simple model names used in UI/API (e.g. "gpt-4o")
# to the exact keys in BASE_PRICING_PER_MILLION_TOKENS.
SIMPLE_NAME_TO_PRICING_KEY_MAP = {
    "gpt-4o":                           "gpt-4o-2024-11-20",
    "gpt-4-turbo":                      "gpt-4-turbo",
    "gpt-3.5-turbo":                    "gpt-3.5-turbo",
    "chatgpt-4o-latest":                "chatgpt-4o-latest",
    "o4-mini":                          "o4-mini",
    "o3":                               "o3",
    "o1-pro":                           "o1-pro",
    "gpt-4.1":                          "gpt-4.1",
    "gpt-4.1-mini":                     "gpt-4.1-mini",
    "gpt-4.1-nano":                     "gpt-4.1-nano",
    "claude-3-opus":                    "claude-3-opus-20240229",
    "claude-3-sonnet":                  "claude-3-5-sonnet-20241022", # Mapping simple name to the latest specific sonnet
    "claude-3-haiku":                   "claude-3-haiku-20240307",
    "claude-3-7-sonnet-20250219":       "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-thinking":       "claude-3-7-sonnet-thinking",
    "claude-3-5-sonnet-20241022":       "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022":        "claude-3-5-haiku-20241022",
    "gemini-pro":                       "gemini-2.5-pro-exp-03-25", # Mapping simple name to the specific pro
    "gemini-1.5-flash-latest":          "gemini-1.5-flash-latest",
    "gemini-2.5-flash":                 "gemini-2.5-flash",
    "gemini-2.5-flash-thinking":        "gemini-2.5-flash-thinking",
    "gemini-2.0-flash-lite-preview":    "gemini-2.0-flash-lite-preview",
    "gemini-1.5-flash-8b":              "gemini-1.5-flash-8b",
    "grok-3-beta":                      "grok-3-beta",
    "grok-3-fast-beta":                 "grok-3-fast-beta",
    "grok-3-mini-beta":                 "grok-3-mini-beta",
    "grok-3-mini-fast-beta":            "grok-3-mini-fast-beta",
    "grok-3-mini":                           "grok-3-mini-beta", # Mapping simple name 'grok-2'
    "deepseek-chat":                    "deepseek-chat",
    "deepseek-reasoner":                "deepseek-reasoner",
    "DeepSeek-R1-Zero":                 "DeepSeek-R1-Zero",
    "DeepSeek-R1":                      "DeepSeek-R1",
}

DEFAULT_BEST_OF_N = 8 # Same default as used in JS and Python simulator

def get_pricing_info(model_id: str) -> dict:
    """
    Retrieve pricing info and Best-of-N for a given model ID.

    Args:
        model_id (str): E.g. "gpt-4o-B5" or "claude-3-opus"

    Returns:
        dict: Contains 'input', 'output', 'category', and 'bestOfN' keys.
              Returns {'input': 0, 'output': 0, 'category': 'Unknown', 'bestOfN': n}
              if pricing info is not found.
    """
    parts = model_id.split("-B")
    base_name = parts[0]
    best_of_n = 1 # Default if no -B suffix

    if len(parts) > 1:
        n_str = parts[1]
        if n_str.isdigit():
            best_of_n = int(n_str)
        elif n_str == "": # Case like "model-B"
            best_of_n = DEFAULT_BEST_OF_N
        else:
            # Handle invalid N suffix if necessary, or let it default to DEFAULT_BEST_OF_N
            # print(f"Warning: Invalid Best-of-N suffix '{n_str}' in model ID '{model_id}'. Using default N={DEFAULT_BEST_OF_N}.")
            best_of_n = DEFAULT_BEST_OF_N # Defaulting might be safer

        # Ensure N is at least 1
        if best_of_n < 1:
            best_of_n = 1

    # Find the specific pricing key using the map
    pricing_key = SIMPLE_NAME_TO_PRICING_KEY_MAP.get(base_name)

    # Get the base prices using the specific key
    base_prices = BASE_PRICING_PER_MILLION_TOKENS.get(pricing_key) if pricing_key else None

    if not base_prices:
        # Use print for warning to match JS behavior, replace with logging.warning in production
        print(f'Warning: Pricing not found for model "{model_id}" (mapped key: "{pricing_key}"). Returning zero prices.')
        # Return a default structure with bestOfN included
        return { "input": 0, "output": 0, "category": "Unknown", "bestOfN": best_of_n }

    # Return a copy of the base prices dictionary merged with the bestOfN value
    # Using dict unpacking {**base_prices} creates a shallow copy
    return { **base_prices, "bestOfN": best_of_n }

# Example Usage (optional - for testing this file directly)
if __name__ == "__main__":
    test_models = [
        "gpt-4o-B5",
        "gpt-4o",
        "claude-3-opus-B3",
        "grok-2-B10", # Uses mapping for 'grok-2'
        "gemini-pro", # No -B
        "deepseek-chat-B", # Uses default N
        "gpt-4.1-mini-B1", # N=1 case
        "non_existent_model-B7",
        "gpt-4o-BInvalid", # Should default N
        "claude-3-5-sonnet-20241022",
        "claude-3-sonnet" # Mapped name
    ]

    for model in test_models:
        info = get_pricing_info(model)
        print(f"Model: {model:<35} -> Pricing Info: {info}")