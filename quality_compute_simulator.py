# quality_compute_simulator.py
# -*- coding: utf-8 -*-
import re
import os
import uuid
import json
import concurrent.futures
from functools import wraps
import traceback
import sqlite3
import bcrypt
from datetime import datetime, timedelta
import jwt
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests  # For xAI, Anthropic, and other API calls if needed
from openai import OpenAI  # For OpenAI and DeepSeek API calls
from google import genai
import smtplib
from email.mime.text import MIMEText
from pricing import get_pricing_info  # Import pricing info
import math  # For math operations in cost estimation
from flask_cors import CORS
from auth import auth_bp  # Import the blueprint from auth.py
from payments import payments_bp  # New import for payments blueprint
from utils import send_email_notification
load_dotenv()

app = Flask(__name__)
app.register_blueprint(auth_bp)  # Register the blueprint with the app
app.register_blueprint(payments_bp, url_prefix='/payments')  # New registration with URL prefix
CORS(app, resources={r"/*": {
                        "origins": ["http://localhost:3000", "https://qualitycompute.henosis.us"],
                        "methods": ["GET", "POST", "OPTIONS"],
                        "allow_headers": ["Content-Type", "Authorization"], # Correct way to specify allowed headers per resource
                        "supports_credentials": True
                    }
         },
     supports_credentials=True)
print("--- [DEBUG CORS] CORS Configuration Details ---")
print("--- [DEBUG CORS] Resources: ", app.config.get('CORS_RESOURCES', {}))
print("--- [DEBUG CORS] Supports Credentials: ", app.config.get('CORS_SUPPORTS_CREDENTIALS', False))
# --- Configuration ---
MODEL_API_BASE_URL = os.environ.get("MODEL_API_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-nano")  # Fallback if not specified in request or defaulting fails
DEFAULT_N_VALUE = 8
MAX_WORKERS = 2048
# Import JWT secret from environment (ensure it's consistent with auth.py)
JWT_SECRET = os.environ.get("JWT_SECRET", "your_jwt_secret_here")  # Replace with secure secret in production
DB_PATH = os.environ.get("DB_PATH", "quality_compute.db")

# JWT Verification Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header with Bearer token is required"}), 401
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = payload.get('user_id')  # Extract user_id from JWT payload
            if not user_id:
                return jsonify({"error": "Invalid token: missing user_id"}), 401
            return f(user_id, *args, **kwargs)  # Pass user_id to the endpoint function
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"error": "Token verification failed"}), 401
    return decorated
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")  # Admin password for secured endpoint
# Helper function for granularity group by (unchanged)
def get_granularity_group_by(granularity):
    if granularity == "hourly":
        return "HOUR(timestamp) as time_bucket, DATE(timestamp)"
    elif granularity == "daily":
        return "DATE(timestamp) as time_bucket"
    elif granularity == "weekly":
        return "STRFTIME('%Y-%W', timestamp) as time_bucket"
    elif granularity == "monthly":
        return "STRFTIME('%Y-%m', timestamp) as time_bucket"
    else:
        return "DATE(timestamp) as time_bucket"  # Default to daily
# Helper function to get time bucket expression for SELECT and GROUP BY
def get_time_bucket_expression(granularity):
    if granularity == "hourly":
        return "STRFTIME('%Y-%m-%d %H:00', timestamp) as time_bucket"
    elif granularity == "daily":
        return "STRFTIME('%Y-%m-%d', timestamp) as time_bucket"
    elif granularity == "weekly":
        return "STRFTIME('%Y-%W', timestamp) as time_bucket"  # ISO week number
    elif granularity == "monthly":
        return "STRFTIME('%Y-%m', timestamp) as time_bucket"
    else:
        return "STRFTIME('%Y-%m-%d', timestamp) as time_bucket"  # Default to daily

# Helper function for GROUP BY clause (uses the alias 'time_bucket')
def get_group_by_clause(group_by):
    if group_by == "api_key":
        return "time_bucket, api_key"
    elif group_by == "model":
        return "time_bucket, model_used"
    elif group_by == "token_type":
        return "time_bucket, token_type"  # For token_type, GROUP BY is handled in the UNION query
    else:
        return "time_bucket"  # Default    
# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("--- WARNING: OPENAI_API_KEY environment variable not set. OpenAI provider may not function. ---")
    OPENAI_API_KEY = "dummy-openai-key"  # Use a placeholder to avoid crashing client init

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=MODEL_API_BASE_URL,
    api_key=OPENAI_API_KEY
)
print(f"--- OpenAI client initialized with Service API Key (ending in ...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 4 else '****'}) ---")

# xAI Configuration
XAI_API_BASE_URL = os.environ.get("XAI_API_BASE_URL", "https://api.x.ai")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
if not XAI_API_KEY:
    print("--- WARNING: XAI_API_KEY environment variable not set. xAI provider will not function without it. ---")

# Anthropic Configuration (No client initialization; handled manually with requests)
ANTHROPIC_API_BASE_URL = os.environ.get("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("--- WARNING: ANTHROPIC_API_KEY environment variable not set. Anthropic provider will not function without it. ---")

# DeepSeek Configuration
DEEPSEEK_API_BASE_URL = os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        deepseek_client = OpenAI(  # Use OpenAI client since DeepSeek API is compatible
            base_url=DEEPSEEK_API_BASE_URL,
            api_key=DEEPSEEK_API_KEY
        )
        print(f"--- DeepSeek client initialized with Service API Key (ending in ...{DEEPSEEK_API_KEY[-4:] if len(DEEPSEEK_API_KEY) > 4 else '****'}) ---")
    except Exception as e:
        print(f"--- ERROR: Failed to initialize DeepSeek client: {e} ---")
        deepseek_client = None
else:
    print("--- WARNING: DEEPSEEK_API_KEY environment variable not set. DeepSeek provider will not function without it. ---")

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"--- Gemini API configured with Service API Key (ending in ...{GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 4 else '****'}) ---")
    except Exception as e:
        print(f"--- ERROR: Failed to configure Gemini API: {e} ---")
else:
    print("--- WARNING: GEMINI_API_KEY environment variable not set. Gemini provider will not function without it. ---")

# --- Database Initialization ---
def init_db():
    """Initialize the SQLite database with necessary tables if they don't exist, and ensure columns are present."""
    conn = None  # Initialize conn to None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create users table if not exists with basic columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                credits REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create payment_records table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_records (
                payment_intent_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                credits_added REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Create API keys table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Create API logs table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_logs (
                log_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                api_key TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT NOT NULL,
                n_calls INTEGER NOT NULL,
                input_data TEXT NOT NULL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                reasoning_tokens INTEGER,
                candidate_responses TEXT,
                final_response TEXT,
                evaluator_model TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Check existing columns in the users table
        cursor.execute("PRAGMA table_info(users)")
        existing_columns_info = cursor.fetchall()
        existing_column_names = {col[1] for col in existing_columns_info}

        # List of all columns that should exist, with their definitions and default values
        columns_to_ensure = [
            ('email_verified', 'BOOLEAN DEFAULT 0', 0),  # 0 for false
            ('verification_code', 'TEXT', None),         # No default, can be NULL
            ('verification_code_expires_at', 'TIMESTAMP', None),  # No default, can be NULL
            ('organization_name', 'TEXT DEFAULT NULL', None),
            ('email_alerts', 'BOOLEAN DEFAULT 1', 1),
            ('low_credit_threshold', 'REAL DEFAULT 2.0', 2.0),
            ('usage_reports', "TEXT DEFAULT 'never'", 'never'),
            # New columns for password reset functionality
            ('reset_password_token', 'TEXT DEFAULT NULL', None),
            ('reset_password_token_expires_at', 'TIMESTAMP DEFAULT NULL', None)
        ]

        # Add missing columns and set their default values for existing rows
        for col_name, col_definition, default_value in columns_to_ensure:
            if col_name not in existing_column_names:
                try:
                    alter_query = f"ALTER TABLE users ADD COLUMN {col_name} {col_definition}"
                    print(f"--- Executing: {alter_query} ---")
                    cursor.execute(alter_query)

                    # If the column has a default value, set it for existing rows
                    if default_value is not None:
                        update_query = f"UPDATE users SET {col_name} = ? WHERE {col_name} IS NULL"
                        print(f"--- Executing: {update_query} with value: {default_value} (type: {type(default_value)}) ---")
                        cursor.execute(update_query, (default_value,))
                    conn.commit()
                    print(f"--- Added '{col_name}' column to users table and set default values if applicable ---")

                    # Special migration for email_verified if it was just added (as per existing code)
                    if col_name == 'email_verified':
                        cursor.execute("UPDATE users SET email_verified = 1")
                        conn.commit()
                        print("--- Migration: Set email_verified to 1 for all existing users to avoid login issues ---")
                except sqlite3.Error as e:
                    print(f"--- ERROR adding column '{col_name}' or setting default: {e} ---")
                    traceback.print_exc()
                    if conn:
                        conn.rollback()
            else:
                print(f"--- '{col_name}' column already exists in users table ---")

        conn.commit()  # Final commit for any changes
        print(f"--- Database initialized/updated at {DB_PATH} ---")

    except sqlite3.Error as e:
        print(f"--- CRITICAL DATABASE ERROR during init_db: {e} ---")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("--- Database connection closed ---")
init_db()


def strip_best_of_suffix(model_id):
    """
    Strip the -B(num) suffix from a model ID if present.
    For example, 'gemini-2.5-flash-preview-04-17-B8' becomes 'gemini-2.5-flash-preview-04-17'.
    Args:
        model_id (str): The model ID string.
    Returns:
        str: The model ID with -B(num) suffix removed if it matches the pattern.
    """
    match = re.match(r'^(.+?)(-B(\d*))?$', model_id)
    if match:
        return match.group(1)  # Return the base model part
    else:
        return model_id  # If no -B suffix, return the original model ID
# --- Helper Function to Log API Calls ---
def log_api_call(user_id, api_key, model_used, n_calls, input_data, usage_info, candidate_responses, final_response, evaluator_model):
    log_id = str(uuid.uuid4())
    timestamp = datetime.now()
    tokens_in = usage_info.get('input_tokens', 0) if usage_info else 0
    tokens_out = usage_info.get('output_tokens', 0) if usage_info else 0
    reasoning_tokens = usage_info.get('reasoning_tokens', 0) if usage_info else 0

    try:
        input_data_json = json.dumps(input_data)
        candidates_json = json.dumps(candidate_responses)
        final_response_str = str(final_response) if final_response is not None else ""

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO api_logs (
                log_id, user_id, api_key, timestamp, model_used, n_calls, input_data,
                tokens_in, tokens_out, reasoning_tokens, candidate_responses,
                final_response, evaluator_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_id, user_id, api_key, timestamp, model_used, n_calls, input_data_json,
            tokens_in, tokens_out, reasoning_tokens, candidates_json,
            final_response_str, evaluator_model
        ))
        conn.commit()
        conn.close()
        print(f"--- API call logged successfully for user {user_id}, API key {api_key[:4]}...{api_key[-4:]} (Log ID: {log_id}) ---")
    except Exception as e:
        print(f"--- CRITICAL ERROR: Failed to log API call for user {user_id}: {e} ---")
        traceback.print_exc()

# --- Helper Function to Call Model API ---
def call_model_api(model_id, input_data, **passthrough_params):
    """
    Calls the configured model API endpoint based on the provider.
    Determines provider from model_id: 'grok-' prefix for xAI, 'claude-' for Anthropic, 
    'deepseek-' for DeepSeek, 'gemini-' for Gemini, else OpenAI.

    Args:
        model_id (str): The specific model ID.
        input_data (str or list): The input prompt or structured input (messages).
        **passthrough_params: Additional parameters for the underlying API.

    Returns:
        tuple: (text_content, usage_dict, error_info)
            - text_content: The response text or None if error.
            - usage_dict: A dictionary with usage info or None if error.
            - error_info: None on success, or a dict {'status_code': int, 'error_detail': str} on error.
    """
    # Strip any -B(num) suffix from the model ID to ensure valid API calls
    original_model_id = model_id  # Keep original for logging
    model_id = strip_best_of_suffix(model_id)  # Apply stripping here
    
    print(f"--- Calling Provider API with Original Model: {original_model_id}, Stripped Model: {model_id}, Provider: to be determined ---")
    
    # Handle input type logging and formatting
    messages = []
    if isinstance(input_data, str):
        print(f"--- Input (String): {input_data[:100]}... ---")
        messages = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        print(f"--- Input (List): {len(input_data)} messages ---")
        messages = input_data  # Assume it's already in message format
    else:
        print(f"--- ERROR: Invalid input type for API call. Must be string or list. ---")
        return None, None, {"status_code": 400, "error_detail": "Invalid input type for API call. Must be string or list."}

    # Prepare parameters - remove provider-specific unsupported params later
    api_params = passthrough_params.copy()
    if api_params:
        print(f"--- Passthrough Parameters (Initial): {api_params} ---")

    # Determine provider based on stripped model_id
    provider = "openai"  # Default
    if model_id.startswith("grok-"):
        provider = "xai"
    elif model_id.startswith("claude-"):
        provider = "anthropic"
    elif model_id.startswith("deepseek-"):
        provider = "deepseek"
    elif model_id.startswith("gemini-"):  # Use the stripped model_id for provider detection
        provider = "gemini"

    print(f"--- Final Provider: {provider} with Stripped Model: {model_id} ---")

    # --- OpenAI Provider ---
    if provider == "openai":
        local_client = openai_client  # Use the initialized client
        # Handle OpenAI-specific parameter adjustments (e.g., reasoning object)
        if "reasoning" in api_params and isinstance(api_params["reasoning"], str):
            reasoning_value = api_params["reasoning"]
            api_params["reasoning"] = {"effort": reasoning_value}
            print(f"--- Converted 'reasoning' string '{reasoning_value}' to object for OpenAI ---")
        elif "reasoning_effort" in api_params:
            print(f"--- Translating 'reasoning_effort' to OpenAI 'reasoning' object ---")
            api_params["reasoning"] = {"effort": api_params.pop("reasoning_effort")}
        # Remove params not supported by OpenAI Chat Completion API if necessary
        api_params.pop("reasoning_effort", None)
        api_params.pop("thinking", None)  # Remove Anthropic-specific params
        api_params.pop("betas", None)      # Remove Anthropic-specific params
        api_params.pop("thinkingBudget", None)  # Remove Gemini-specific params
        # Map max_output_tokens or max_completion_tokens to max_tokens
        if "max_output_tokens" in api_params:
            api_params["max_tokens"] = api_params.pop("max_output_tokens")
        elif "max_completion_tokens" in api_params:  # Handle the 'max_completion_tokens' param from your log
            api_params["max_completion_tokens"] = api_params.pop("max_completion_tokens")

        print(f"--- Final Params for OpenAI: {api_params} ---")
        try:
            response = local_client.chat.completions.create(
                model=model_id,  # Use stripped model_id
                messages=messages,
                **api_params
            )
            
            # Add detailed logging for debugging response structure
            print(f"--- OpenAI Full Response Object for model {model_id}: {response} ---")
            if hasattr(response, 'choices'):
                print(f"--- OpenAI Response Choices: {len(response.choices)} choices found ---")
                for choice in response.choices:
                    print(f"--- Choice Message: {choice.message} ---")
            else:
                print("--- OpenAI Response has no 'choices' attribute or is empty ---")
            
            if hasattr(response, 'usage'):
                print(f"--- OpenAI Usage Object: {response.usage} ---")
                print(f"--- OpenAI Usage Attributes: {dir(response.usage)} ---")
                print(f"--- OpenAI Usage prompt_tokens: {getattr(response.usage, 'prompt_tokens', 'N/A')} ---")
                print(f"--- OpenAI Usage completion_tokens: {getattr(response.usage, 'completion_tokens', 'N/A')} ---")
                print(f"--- OpenAI Usage total_tokens: {getattr(response.usage, 'total_tokens', 'N/A')} ---")
                # Check for reasoning_tokens in completion_tokens_details
                if hasattr(response.usage, 'completion_tokens_details') and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    print(f"--- OpenAI Usage reasoning_tokens found in completion_tokens_details: {getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 'N/A')} ---")
                else:
                    print("--- OpenAI Usage does not have 'reasoning_tokens' in completion_tokens_details ---")
                    # Check for any other potential reasoning-related fields
                    for attr in dir(response.usage):
                        if 'reasoning' in attr.lower():
                            print(f"--- OpenAI Usage has potential reasoning-related attribute '{attr}': {getattr(response.usage, attr, 'N/A')} ---")
            else:
                print("--- OpenAI Response has no 'usage' attribute ---")
            
            # Extract text content with robust attribute checks
            text_content = None
            if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                text_content = response.choices[0].message.content
            else:
                print(f"--- OpenAI API Call Warning: Could not extract text content from response for {model_id}. ---")
            
            # Extract usage with proper handling of reasoning_tokens
            usage_dict = {
                'input_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                'output_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                'total_tokens': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                'reasoning_tokens': 0  # Initialize to 0, then try to extract from nested attribute
            }
            if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details') and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                usage_dict['reasoning_tokens'] = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
                print(f"--- Extracted reasoning_tokens: {usage_dict['reasoning_tokens']} ---")
            else:
                print("--- Could not extract reasoning_tokens; set to 0 ---")
            
            if text_content:
                print(f"--- OpenAI API Call Successful for {model_id} ---")
                print(f"--- Response Text (Snippet): {text_content[:100]}... ---")
            return text_content, usage_dict, None  # Success, no error

        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "incorrect api key" in error_msg:
                print(f"--- CRITICAL ERROR: OpenAI API Authentication Failed for {model_id}. Check OPENAI_API_KEY. ---")
            else:
                print(f"--- Error during OpenAI API call for {model_id}: {e} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(e)}

    # --- xAI Provider ---
    elif provider == "xai":
        if not XAI_API_KEY:
            print("--- CRITICAL ERROR: XAI_API_KEY not set. Cannot call xAI API. ---")
            return None, None, {"status_code": 500, "error_detail": "XAI_API_KEY not set."}

        # Map parameters to xAI format
        body = {"model": model_id, "messages": messages}  # Use stripped model_id
        if "max_output_tokens" in api_params:
            body["max_tokens"] = api_params.pop("max_output_tokens")  # xAI uses max_tokens
        if "temperature" in api_params:
            body["temperature"] = api_params["temperature"]

        # Handle reasoning for xAI only if model contains "mini" (case-insensitive)
        if "mini" in model_id.lower():
            reasoning_effort = "low"  # Default
            if "reasoning" in api_params:
                if isinstance(api_params["reasoning"], dict) and "effort" in api_params["reasoning"]:
                    reasoning_effort = api_params["reasoning"]["effort"]
                elif isinstance(api_params["reasoning"], str):
                    reasoning_effort = api_params["reasoning"]
                print(f"--- Using reasoning_effort '{reasoning_effort}' for xAI model {model_id} ---")
            elif "reasoning_effort" in api_params:
                reasoning_effort = api_params["reasoning_effort"]
                print(f"--- Using direct reasoning_effort '{reasoning_effort}' for xAI model {model_id} ---")
            body["reasoning_effort"] = reasoning_effort
        else:
            print(f"--- Skipping reasoning_effort for xAI model {model_id} as it may not be supported ---")
            # Clean up parameters to avoid any potential issues
            api_params.pop("reasoning_effort", None)
            api_params.pop("reasoning", None)
        # Remove Anthropic-specific and Gemini-specific params
        api_params.pop("thinking", None)
        api_params.pop("betas", None)
        api_params.pop("thinkingBudget", None)

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",  # Note: xAI might use Bearer token, confirm if needed
            "Content-Type": "application/json",
        }
        print(f"--- Final Payload for xAI: {body} ---")
        try:
            response = requests.post(f"{XAI_API_BASE_URL}/v1/chat/completions", headers=headers, json=body)
            response.raise_for_status()
            resp_json = response.json()

            # Add detailed logging here to inspect the full response
            print(f"--- xAI Full Response JSON for model {model_id}: {json.dumps(resp_json, indent=2)} ---")
            
            # Extract and log the usage dictionary specifically
            usage_dict_raw = resp_json.get("usage", {})
            print(f"--- xAI Raw Usage Dictionary for model {model_id}: {usage_dict_raw} ---")

            text_content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "") if resp_json.get("choices") else None
            usage_dict = resp_json.get("usage", {})

            # Updated usage_return with correct key mapping for xAI
            usage_return = {
                'input_tokens': usage_dict.get("prompt_tokens", 0),  # Map xAI's "prompt_tokens" to "input_tokens"
                'output_tokens': usage_dict.get("completion_tokens", 0),  # Map xAI's "completion_tokens" to "output_tokens"
                'total_tokens': usage_dict.get("total_tokens", 0),
                'reasoning_tokens': usage_dict.get("completion_tokens_details", {}).get("reasoning_tokens", 0)  # Extract from nested dict
            }

            # Add logging after parsing to verify the fix
            print(f"--- xAI Parsed Usage Return for model {model_id}: {usage_return} ---")  # Log the parsed usage values

            if text_content:
                print(f"--- xAI API Call Successful for {model_id} ---")
                print(f"--- Response Text (Snippet): {text_content[:100]}... ---")
            else:
                print(f"--- xAI API Call Warning: Could not extract text from response for {model_id}. ---")
            return text_content, usage_return, None  # Success, no error

        except requests.exceptions.HTTPError as http_err:
            print(f"--- HTTP Error during xAI API call for {model_id}: {http_err} ---")
            try:
                print(f"--- xAI Error Response Body: {response.text} ---")
                if response.status_code == 401:
                    print("--- CRITICAL ERROR: xAI API Authentication Failed. Check XAI_API_KEY. ---")
            except Exception:
                pass
            traceback.print_exc()
            return None, None, {"status_code": response.status_code, "error_detail": str(http_err)}
        except Exception as e:
            print(f"--- Error during xAI API call for {model_id}: {e} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(e)}

    # --- Anthropic Provider ---
    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            print("--- CRITICAL ERROR: ANTHROPIC_API_KEY not set. Cannot call Anthropic API. ---")
            return None, None, {"status_code": 500, "error_detail": "ANTHROPIC_API_KEY not set."}

        # Prepare headers with manual x-api-key
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",  # Set API version manually; adjust as needed
            "Content-Type": "application/json"
        }

        # Prepare body for Anthropic Messages API
        body = {
            "model": model_id,  # Use stripped model_id
            "max_tokens": api_params.get("max_tokens", 1024),  # Default to 1024 if not provided
            "messages": messages,
        }

        # Add other parameters if present
        if "temperature" in api_params:
            body["temperature"] = api_params["temperature"]
        if "system" in api_params:
            body["system"] = api_params["system"]
        if "stop_sequences" in api_params:
            body["stop_sequences"] = api_params["stop_sequences"]
        # Remove unsupported parameters
        api_params.pop("thinking", None)
        api_params.pop("betas", None)
        api_params.pop("reasoning", None)
        api_params.pop("reasoning_effort", None)
        api_params.pop("thinkingBudget", None)

        print(f"--- Final Params for Anthropic: {body} ---")
        try:
            response = requests.post(f"{ANTHROPIC_API_BASE_URL}/v1/messages", headers=headers, json=body)
            response.raise_for_status()
            resp_json = response.json()

            # Extract text content
            text_content = None
            if resp_json.get("content") and isinstance(resp_json["content"], list) and len(resp_json["content"]) > 0:
                first_block = resp_json["content"][0]
                if first_block.get("type") == "text":
                    text_content = first_block.get("text")

            # Extract usage
            usage_dict = resp_json.get("usage", {})
            usage_return = {
                'input_tokens': usage_dict.get("input_tokens", 0),
                'output_tokens': usage_dict.get("output_tokens", 0),
                'total_tokens': usage_dict.get("input_tokens", 0) + usage_dict.get("output_tokens", 0),  # Calculate total
                'reasoning_tokens': 0  # Anthropic API doesn't return this
            }

            if text_content:
                print(f"--- Anthropic API Call Successful for {model_id} ---")
                print(f"--- Response Text (Snippet): {text_content[:100]}... ---")
            else:
                print(f"--- Anthropic API Call Warning: Could not extract text content from response for {model_id}. Response: {resp_json} ---")
            return text_content, usage_return, None  # Success, no error

        except requests.exceptions.HTTPError as http_err:
            print(f"--- HTTP Error during Anthropic API call for {model_id}: {http_err} ---")
            print(f"--- Response text: {response.text}")
            error_detail = "Unknown error"
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", str(http_err))
            except Exception:
                error_detail = str(http_err)  # Fallback if JSON parsing fails
            if response.status_code == 400:
                print(f"--- Client-side error (400): Passing through to client: {error_detail} ---")
                return None, None, {"status_code": response.status_code, "error_detail": error_detail}
            elif response.status_code == 401:
                print("--- CRITICAL ERROR: Anthropic API Authentication Failed. Check ANTHROPIC_API_KEY. ---")
                return None, None, {"status_code": 401, "error_detail": "Authentication failure with Anthropic API"}
            traceback.print_exc()
            return None, None, {"status_code": response.status_code, "error_detail": error_detail}
        except Exception as e:
            print(f"--- Error during Anthropic API call for {model_id}: {e} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(e)}

    # --- DeepSeek Provider ---
    elif provider == "deepseek":
        if not deepseek_client:
            print("--- CRITICAL ERROR: DeepSeek client not initialized or DEEPSEEK_API_KEY not set. Cannot call DeepSeek API. ---")
            return None, None, {"status_code": 500, "error_detail": "DeepSeek API key not set or initialization failed."}

        local_client = deepseek_client  # Use the initialized client (OpenAI-compatible)

        # Map parameters to DeepSeek format (similar to OpenAI)
        deepseek_params = api_params.copy()
        # Remove unsupported parameters (e.g., Anthropic-specific or xAI-specific or Gemini-specific)
        deepseek_params.pop("thinking", None)
        deepseek_params.pop("betas", None)
        deepseek_params.pop("reasoning_effort", None)
        deepseek_params.pop("reasoning", None)
        deepseek_params.pop("thinkingBudget", None)  # Remove Gemini-specific params
        # Map max_output_tokens to max_tokens if present
        if "max_output_tokens" in deepseek_params:
            deepseek_params["max_tokens"] = deepseek_params.pop("max_output_tokens")

        print(f"--- Final Params for DeepSeek: {deepseek_params} ---")
        try:
            response = local_client.chat.completions.create(
                model=model_id,  # Use stripped model_id
                messages=messages,
                **deepseek_params
            )
            # Extract main content and reasoning if available
            text_content = response.choices[0].message.content if response.choices and response.choices[0].message else None
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None) if response.choices and response.choices[0].message else None  # Optional, for logging or future use
            # For now, return only the main content in text_content; reasoning is not part of the standard response
            usage_dict = {
                'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'output_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
                'reasoning_tokens': 0  # DeepSeek may not provide this; set to 0 for consistency
            }
            if text_content:
                print(f"--- DeepSeek API Call Successful for {model_id} ---")
                print(f"--- Response Text (Snippet): {text_content[:100]}... ---")
                if reasoning_content:
                    print(f"--- Reasoning Content (Snippet): {reasoning_content[:100]}... ---")  # Optional logging
            else:
                print(f"--- DeepSeek API Call Warning: Could not extract text from response for {model_id}. ---")
            return text_content, usage_dict, None  # Success, no error

        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "incorrect api key" in error_msg:
                print(f"--- CRITICAL ERROR: DeepSeek API Authentication Failed for {model_id}. Check DEEPSEEK_API_KEY. ---")
            else:
                print(f"--- Error during DeepSeek API call for {model_id}: {e} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(e)}

    # --- Gemini Provider ---
    elif provider == "gemini":
        if not genai:  # Check if genai is configured properly
            print("--- CRITICAL ERROR: Gemini API not configured. Check GEMINI_API_KEY. ---")
            return None, None, {"status_code": 500, "error_detail": "Gemini API not configured."}

        # Map parameters to Gemini format
        generation_config = GenerationConfig(
            temperature=api_params.get("temperature"),  # Default temperature if not specified
            max_output_tokens=api_params.get("max_output_tokens"),  # Map from max_output_tokens or use default
            top_p=api_params.get("top_p"),  # Pass through if present
            top_k=api_params.get("top_k"),  # Pass through if present
        )
        # Handle thinking_budget if applicable
        if "thinkingBudget" in api_params and model_id == "gemini-2.5-flash-preview-04-17":
            generation_config.thinking_budget = api_params.pop("thinkingBudget")
        elif "thinkingBudget" in api_params:
            print(f"--- Warning: 'thinkingBudget' ignored for unsupported model {model_id} ---")
            api_params.pop("thinkingBudget", None)

        # Remove other unsupported parameters
        for key in ["reasoning", "reasoning_effort", "thinking", "betas"]:
            api_params.pop(key, None)

        # Handle safety settings if provided (optional, can be added)
        safety_settings = []

        # Handle system_instruction if provided
        system_instruction = api_params.pop("system_instruction", None)

        # Format contents for Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model" if msg["role"] == "assistant" else msg["role"]
            parts = [{"text": msg["content"]}]
            contents.append({"role": role, "parts": parts})

        print(f"--- Final Params for Gemini: generation_config={generation_config}, safety_settings={safety_settings}, contents={contents} ---")

        try:
            model = genai.GenerativeModel(model_name=model_id)  # Use stripped model_id
            response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings if safety_settings else None)

            # Extract text content
            text_content = None
            if hasattr(response, 'text'):
                text_content = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                first_candidate = response.candidates[0]
                if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts') and first_candidate.content.parts:
                    text_content = first_candidate.content.parts[0].text if hasattr(first_candidate.content.parts[0], 'text') else None

            # Manual token counting for Gemini (since usage_metadata might not be reliable)
            input_token_count = 0
            output_token_count = 0
            total_token_count = 0

            try:
                input_token_count = model.count_tokens(contents).total_tokens
                print(f"--- Gemini Manual Input Token Count: {input_token_count} ---")
            except Exception as count_err:
                print(f"--- Error counting input tokens with model.count_tokens: {count_err} ---")
                traceback.print_exc()

            try:
                if text_content:
                    output_token_count = model.count_tokens(text_content).total_tokens
                    print(f"--- Gemini Manual Output Token Count: {output_token_count} ---")
                else:
                    print("--- Gemini: No text_content received, output tokens set to 0. ---")
                    output_token_count = 0
            except Exception as count_err:
                print(f"--- Error counting output tokens with model.count_tokens: {count_err} ---")
                traceback.print_exc()

            total_token_count = input_token_count + output_token_count

            # Populate usage_dict with manually counted tokens
            usage_dict = {
                'input_tokens': input_token_count,
                'output_tokens': output_token_count,
                'reasoning_tokens': 0,  # Gemini does not provide reasoning tokens
                'total_tokens': total_token_count
            }

            if text_content:
                print(f"--- Gemini API Call Successful for {model_id} ---")
                print(f"--- Response Text (Snippet): {text_content[:100]}... ---")
            else:
                print(f"--- Gemini API Call Warning: Could not extract text from response for {model_id}. ---")

            return text_content, usage_dict, None  # Success, no error

        except genai.errors.BlockedPromptException as bpe:
            print(f"--- Gemini API Error for {model_id}: Prompt Blocked - {bpe} ---")
            print(f"--- Prompt Feedback (if available): {getattr(bpe, 'prompt_feedback', 'N/A')} ---")
            input_token_count_err = 0
            try:
                input_token_count_err = model.count_tokens(contents).total_tokens
            except Exception:
                pass
            return None, {'input_tokens': input_token_count_err, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': input_token_count_err}, {"status_code": 400, "error_detail": str(bpe)}
        except genai.errors.StopCandidateException as sce:
            print(f"--- Gemini API Error for {model_id}: Candidate Stopped - {sce} ---")
            input_token_count_err = 0
            try:
                input_token_count_err = model.count_tokens(contents).total_tokens
            except Exception:
                pass
            return None, {'input_tokens': input_token_count_err, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': input_token_count_err}, {"status_code": 400, "error_detail": str(sce)}
        except genai.errors.InvalidArgument as ive:
            print(f"--- Gemini API Invalid Argument Error for {model_id}: {ive} ---")
            print("--- Check model name, parameters, and input format. ---")
            traceback.print_exc()
            return None, None, {"status_code": 400, "error_detail": str(ive)}
        except genai.errors.PermissionDenied as pde:
            print(f"--- Gemini API Permission Denied Error for {model_id}: {pde} ---")
            print("--- CRITICAL ERROR: Check GEMINI_API_KEY and ensure API is enabled. ---")
            traceback.print_exc()
            return None, None, {"status_code": 401, "error_detail": str(pde)}
        except genai.errors.ResourceExhausted as ree:
            print(f"--- Gemini API Resource Exhausted/Rate Limit Error for {model_id}: {ree} ---")
            traceback.print_exc()
            return None, None, {"status_code": 429, "error_detail": str(ree)}
        except genai.errors.InternalServerError as ise:
            print(f"--- Gemini API Internal Server Error for {model_id}: {ise} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(ise)}
        except genai.errors.GoogleAPIError as gae:
            print(f"--- Gemini GoogleAPIError for {model_id}: {gae} ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(gae)}
        except Exception as e:
            print(f"--- Unexpected Error during Gemini API call for {model_id}: {e} (Type: {type(e)}) ---")
            traceback.print_exc()
            return None, None, {"status_code": 500, "error_detail": str(e)}

    else:  # Should not happen with current logic
        print(f"--- ERROR: Unknown provider determined for stripped model {model_id} ---")
        return None, None, {"status_code": 500, "error_detail": f"Unknown provider for model {model_id}"}
# --- Helper Function to Select Best Response using a Judge ---
def select_best_response(original_prompt, candidate_responses, judge_model):
    """Selects best response using the specified judge_model. Uses unique separator."""
    if not candidate_responses:
        print("--- Judge Call Skipped: No candidate responses available ---")
        return None, None, None  # Return None for text, prompt, response_text

    if len(candidate_responses) == 1:
        print("--- Judge Call Skipped: Only one candidate response available ---")
        return candidate_responses[0], "N/A", "N/A"  # Return the single candidate

    judge_prompt_parts = []
    # Format original prompt/conversation
    if isinstance(original_prompt, list):
        judge_prompt_parts.append("Original Conversation:")
        for msg in original_prompt:
            role = msg.get("role", "unknown")
            content = msg.get("content", "<empty>")
            judge_prompt_parts.append(f"{role}: {content}")
    else:  # String input
        judge_prompt_parts.append(f"Original Prompt:\n{original_prompt}")

    # Format candidate responses with unique separator
    judge_prompt_parts.append("\n\nCandidate Responses:")
    separator = "\n=== CANDIDATE SEPARATOR ===\n"
    for i, response in enumerate(candidate_responses):
        judge_prompt_parts.append(f"\n--- Candidate {i+1} ---")
        # Ensure response is a string before appending
        response_str = str(response) if response is not None else "<empty response>"
        judge_prompt_parts.append(response_str)
        if i < len(candidate_responses) - 1:  # Add separator between candidates
            judge_prompt_parts.append(separator)

    judge_prompt_parts.append(
        f"\n\nBased on the original prompt/conversation, which candidate response (1 to {len(candidate_responses)}) is the best? " 
        "Please respond with only the number of the best candidate (e.g., '1', '2', etc.)."
    )
    judge_prompt = "\n".join(judge_prompt_parts)

    print("--- Calling Judge Model ---")
    print(f"--- Judge Model: {judge_model} ---")
    print(f"--- Judge Prompt (Snippet): {judge_prompt[:300]}... ---")

    # Judge call doesn't need complex params, just the prompt
    # Use call_model_api so it works regardless of whether JUDGE_MODEL is OpenAI, xAI, or Anthropic
    judge_response_text, _, judge_error_info = call_model_api(judge_model, judge_prompt)  # Ignore usage for judge call, handle error

    if judge_error_info:
        print(f"--- Error in judge call: Status Code {judge_error_info['status_code']}, Detail: {judge_error_info['error_detail']} ---")
        # Fallback to first candidate if judge fails
        if candidate_responses:
            print("--- Falling back to selecting the first candidate due to judge error. ---")
            selected_text = candidate_responses[0]
        else:
            selected_text = None
        return str(selected_text) if selected_text is not None else None, judge_prompt, judge_error_info

    selected_text = None
    if judge_response_text:
        print(f"--- Judge Raw Response: '{judge_response_text}' ---")
        # More robust extraction of the number
        numbers_found = re.findall(r'\b(\d+)\b', judge_response_text.strip())
        if numbers_found:
            try:
                # Try the first number found
                selected_index = int(numbers_found[0]) - 1
                if 0 <= selected_index < len(candidate_responses):
                    selected_text = candidate_responses[selected_index]
                    print(f"--- Judge selected candidate: {selected_index + 1} ---")
                else:
                    print(f"--- Judge selected invalid index: {selected_index + 1}. Valid range: 1-{len(candidate_responses)} ---")
            except ValueError:
                print("--- Judge response number parsing failed. ---")
        else:
            print("--- Could not find a number in the judge's response. ---")
    else:
        print("--- Judge model failed to provide a response. ---")

    # Fallback to first candidate if judge fails or selects invalid index
    if selected_text is None and candidate_responses:
        print("--- Falling back to selecting the first candidate. ---")
        selected_text = candidate_responses[0]

    # Ensure selected_text is a string before returning
    return str(selected_text) if selected_text is not None else None, judge_prompt, judge_response_text

# --- Helper Function to Validate User API Key ---
def validate_api_key(api_key):
    """Validate USER API key and return user_id if valid, else None."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM api_keys WHERE api_key = ?", (api_key,))
        result = cursor.fetchone()
        if result:
            user_id = result[0]
            return user_id
        else:
            print(f"--- User API Key validation failed: Key '{api_key[:4]}...{api_key[-4:]}' not found ---")
            return None
    except Exception as e:
        print(f"--- Error validating User API key: {e} ---")
        return None
    finally:
        if conn:
            conn.close()

# --- Helper Function for Cost Estimation ---
def estimate_cost(model_id, input_data, n_value, passthrough_params):
    """
    Estimate the cost of an API call based on model pricing and token estimates.
    Uses Gemini for token counting (as per instructions). Estimates output tokens.
    Args:
        model_id (str): The model ID.
        input_data (str or list): The input data.
        n_value (int): The number of calls (for Best-of-N).
        passthrough_params (dict): Additional parameters like max_tokens.
    Returns:
        float: Estimated cost in dollars.
    """
    # Get pricing info
    pricing_info = get_pricing_info(model_id)  # From pricing.py
    if pricing_info["category"] == "Unknown":
        print(f"--- Warning: Pricing not found for model '{model_id}'. Using zero cost estimate. ---")
        return 0.0

    input_price_per_token = pricing_info["input"] / 1000000.0  # Convert per million to per token
    output_price_per_token = pricing_info["output"] / 1000000.0
    reasoning_price_per_token = 0.0  # Assuming no reasoning cost for now; add if needed

    # Estimate input tokens using Gemini (as per instructions; replace with dynamic provider later)
    # Note: This assumes Gemini is configured; handle errors
    try:
        # Convert input_data to Gemini-compatible format
        if isinstance(input_data, str):
            contents = [{"role": "user", "parts": [{"text": input_data}]}]
        elif isinstance(input_data, list):
            contents = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in input_data]
        else:
            raise ValueError("Invalid input type for token estimation")

        # Use Gemini to count input tokens
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")  # Use a default Gemini model
        input_token_count = model.count_tokens(contents).total_tokens
        print(f"--- Estimated input tokens: {input_token_count} ---")
    except Exception as e:
        print(f"--- Error estimating input tokens: {e}. Falling back to len(input_data) as token estimate. ---")
        # Fallback: rough estimate based on character length (1 token ~ 4 chars)
        input_token_count = math.ceil(len(str(input_data)) / 4)

    # Estimate output tokens based on rules
    max_output_estimate = 3000  # Default min output token estimate
    if "max_tokens" in passthrough_params or "max_output_tokens" in passthrough_params:
        max_output_tokens = passthrough_params.get("max_tokens", passthrough_params.get("max_output_tokens", 3000))
        output_token_estimate = min(max_output_tokens, 100000)  # Cap at 100k as per rules
    else:
        input_length_check = input_token_count * 0.5  # 50% of input length
        if input_length_check > 3000:
            output_token_estimate = min(input_length_check, 100000)  # Use 50% of input, cap at 100k
        else:
            output_token_estimate = max_output_estimate  # Min 3k tokens

    # Reason tokens estimate (set to 0 for now; adjust if provider-specific)
    reasoning_token_estimate = 0

    # Calculate cost for one call
    input_cost = input_token_count * input_price_per_token
    output_cost = output_token_estimate * output_price_per_token
    reasoning_cost = reasoning_token_estimate * reasoning_price_per_token
    single_call_cost = input_cost + output_cost + reasoning_cost

    # Multiply by n_value for Best-of-N
    total_estimated_cost = single_call_cost * n_value

    print(f"--- Estimated cost for {model_id} with N={n_value}: ${total_estimated_cost:.6f} ---")
    return total_estimated_cost

# --- Helper Function to Get User Credits ---
def get_user_credits(user_id):
    """
    Fetch the user's current credit balance from the database.
    Args:
        user_id (str): The user's ID.
    Returns:
        float: The user's credits, or 0.0 if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0] is not None:
        return float(result[0])
    return 0.0

# --- Helper Function to Deduct User Credits ---
def deduct_user_credits(user_id, cost):
    """
    Deduct the actual cost from the user's credit balance.
    Args:
        user_id (str): The user's ID.
        cost (float): The cost to deduct.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (cost, user_id))
        conn.commit()
        print(f"--- Deducted ${cost:.6f} from user {user_id} credits ---")
    except Exception as e:
        print(f"--- Error deducting credits for user {user_id}: {e} ---")
        # Optionally, handle rollback or alert
    finally:
        conn.close()

# --- Helper Function to Calculate Actual Cost ---
def calculate_actual_cost(usage, model_id):
    """
    Calculate the actual cost based on usage and model pricing.
    Args:
        usage (dict): Aggregated usage with 'input_tokens', 'output_tokens', etc.
        model_id (str): The model ID for pricing.
    Returns:
        float: Actual cost in dollars.
    """
    pricing_info = get_pricing_info(model_id)
    if pricing_info["category"] == "Unknown":
        print(f"--- Warning: Pricing not found for model '{model_id}'. Using zero cost. ---")
        return 0.0

    input_cost = (usage.get('input_tokens', 0) * (pricing_info["input"] / 1000000.0))
    output_cost = (usage.get('output_tokens', 0) * (pricing_info["output"] / 1000000.0))
    reasoning_cost = (usage.get('reasoning_tokens', 0) * (pricing_info.get("output", 0) / 1000000.0))  # Assuming reasoning uses output price if applicable
    total_cost = input_cost + output_cost + reasoning_cost
    print(f"--- Actual cost for {model_id}: ${total_cost:.6f} ---")
    return total_cost
@app.route('/health')
def health_check():
    """Simple health check endpoint that returns 200 OK.
    Used by Docker healthcheck and nginx to verify service status."""
    app.logger.info("Health check endpoint called")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': os.getenv('VERSION', 'unknown')
    }), 200
#overview endpoint
@app.route('/api/overview', methods=['POST'])
@token_required
def get_overview_data(user_id):
    data = request.get_json()
    time_range = data.get("time_range", "7d")  # Default to 7 days if not specified
    granularity = data.get("granularity", "daily")  # Default granularity
    group_by = data.get("group_by", "model")  # Default group by

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch user details (name and credits)
    cursor.execute("SELECT name, credits FROM users WHERE user_id = ?", (user_id,))
    user_result = cursor.fetchone()
    if not user_result:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    name, credits = user_result

    # Fetch API call counts with fixed 'today' query
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)
    cursor.execute("SELECT COUNT(*) FROM api_logs WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?", (user_id, today_start, today_end))
    today_calls = cursor.fetchone()[0]

    # This week: Use ISO week number
    cursor.execute("SELECT COUNT(*) FROM api_logs WHERE user_id = ? AND strftime('%Y-%W', timestamp) = strftime('%Y-%W', 'now')", (user_id,))
    this_week_calls = cursor.fetchone()[0]

    # Total calls
    cursor.execute("SELECT COUNT(*) FROM api_logs WHERE user_id = ?", (user_id,))
    total_calls = cursor.fetchone()[0]

    # Fetch recent calls (last 5)
    cursor.execute("SELECT log_id, timestamp, model_used, tokens_in, tokens_out, reasoning_tokens FROM api_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (user_id,))
    recent_calls_rows = cursor.fetchall()
    recent_calls = [dict(zip([column[0] for column in cursor.description], row)) for row in recent_calls_rows]

    # Calculate start time based on time_range
    if time_range == "24h":
        start_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    elif time_range == "7d":
        start_time = (datetime.utcnow() - timedelta(days=7)).isoformat()
    else:  # "30d"
        start_time = (datetime.utcnow() - timedelta(days=30)).isoformat()

    # Fetch usage by model
    cursor.execute("""
        SELECT model_used, 
               SUM(tokens_in) as input_tokens, 
               SUM(tokens_out) as output_tokens, 
               SUM(reasoning_tokens) as reasoning_tokens, 
               COUNT(*) as call_count 
        FROM api_logs 
        WHERE user_id = ? AND timestamp >= ? 
        GROUP BY model_used 
        ORDER BY model_used ASC
    """, (user_id, start_time))
    usage_by_model_rows = cursor.fetchall()

    total_input_tokens = sum(row[1] for row in usage_by_model_rows)
    total_output_tokens = sum(row[2] for row in usage_by_model_rows)
    total_reasoning_tokens = sum(row[3] for row in usage_by_model_rows)
    total_tokens = total_input_tokens + total_output_tokens + total_reasoning_tokens

    usage_by_model = []
    for row in usage_by_model_rows:
        model, input_tok, output_tok, reasoning_tok, call_count = row
        percentage = (total_tokens > 0) and ((input_tok + output_tok + reasoning_tok) / total_tokens * 100) or 0.0
        usage_by_model.append({
            "model": model,
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "reasoning_tokens": reasoning_tok,
            "call_count": call_count,
            "percentage": round(percentage, 2)
        })

    # Fetch daily usage
    cursor.execute("""
        SELECT STRFTIME('%Y-%m-%d', timestamp) as date, 
               SUM(tokens_in) as input_tokens, 
               SUM(tokens_out) as output_tokens, 
               SUM(reasoning_tokens) as reasoning_tokens 
        FROM api_logs 
        WHERE user_id = ? AND timestamp >= ? 
        GROUP BY date 
        ORDER BY date ASC
    """, (user_id, start_time))
    daily_usage_rows = cursor.fetchall()
    daily_usage = [dict(zip([column[0] for column in cursor.description], row)) for row in daily_usage_rows]

    conn.close()

    overview_data = {
        "name": name,
        "credits": credits,
        "apiCalls": {
            "today": today_calls,
            "thisWeek": this_week_calls,
            "total": total_calls
        },
        "recentCalls": recent_calls,
        "usageByModel": usage_by_model,
        "dailyUsage": daily_usage
    }

    return jsonify(overview_data)
# --- User Registration Endpoint ---    
@app.route('/api/register', methods=['POST'])
def register_user():
    # Log the receipt of the POST request with timestamp and headers for debugging
    print(f"--- [DEBUG] Received POST request to /api/register at {datetime.now()} ---")
    print(f"--- [DEBUG] Request Headers: {request.headers} ---")
    
    if not request.is_json:
        print("--- [ERROR] Request is not JSON ---")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    print(f"--- [DEBUG] Request Data Received: {data} ---")  # Log the full JSON data
    
    name = data.get("name")
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    
    # Log the extracted data, masking sensitive information like password
    print(f"--- [DEBUG] Extracted Data: name={name}, email={email}, username={username}, password=**** ---")
    
    # Validate required fields
    if not all([name, email, username, password]):
        print("--- [ERROR] Missing required fields in request ---")
        return jsonify({"error": "Name, email, username, and password are required"}), 400
    
    # Generate user ID and hash password
    user_id = str(uuid.uuid4())
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    print(f"--- [DEBUG] Generated user_id: {user_id} and hashed password ---")
    
    # Generate a random 6-digit verification code and expiration (e.g., 24 hours from now)
    import secrets  # Ensure this is imported at the top if not already
    verification_code = str(secrets.randbelow(1000000)).zfill(6)  # 6-digit code
    verification_expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()  # Expires in 24 hours
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print("--- [DEBUG] Attempting to insert user into database ---")
        
        # Insert user with email_verified set to 0 (false), and store verification code and expiration
        cursor.execute(
            "INSERT INTO users (user_id, name, email, username, password, email_verified, verification_code, verification_code_expires_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",  # 0 for email_verified (false)
            (user_id, name, email, username, hashed_password, verification_code, verification_expires_at)
        )
        conn.commit()
        print(f"--- [DEBUG] User inserted successfully: user_id={user_id}, username={username} ---")
        
        # Send verification email using utils.py
        from utils import send_email  # Import send_email function
        email_subject = "Verify Your Email Address"
        email_body = f"Hello {name},\n\nThank you for registering! Please use the following verification code to confirm your email:\n\n{verification_code}\n\nThis code will expire in 24 hours. If you did not request this, please ignore this email.\n\nBest regards,\nQuality Compute Team"
        send_email(email, email_subject, email_body)  # Use send_email from utils.py
        
        return jsonify({"message": "User registered successfully. Please check your email to verify your account.", "user_id": user_id}), 201
    except sqlite3.IntegrityError as e:
        print(f"--- [ERROR] IntegrityError during user insertion: {e} ---")
        if "users.email" in str(e):
            error_msg = "Email already exists"
        elif "users.username" in str(e):
            error_msg = "Username already exists"
        else:
            error_msg = "Email or username already exists"
        return jsonify({"error": error_msg}), 409
    except Exception as e:
        print(f"--- [ERROR] Exception during user registration: {e} ---")
        traceback.print_exc()  # Print full traceback for detailed debugging
        return jsonify({"error": "Failed to register user"}), 500
    finally:
        if conn:
            conn.close()
            print("--- [DEBUG] Database connection closed ---")

# --- New Endpoint for Email Verification ---
@app.route('/api/verify_email', methods=['POST'])
def verify_email():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_id = data.get("user_id")
    verification_code = data.get("verification_code")
    
    if not all([user_id, verification_code]):
        return jsonify({"error": "user_id and verification_code are required"}), 400
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Fetch user record
        cursor.execute("SELECT verification_code, verification_code_expires_at, email_verified FROM users WHERE user_id = ?", (user_id,))
        user_record = cursor.fetchone()
        
        if not user_record:
            return jsonify({"error": "User not found"}), 404
        
        stored_code, expires_at, email_verified = user_record
        
        # Check if already verified
        if email_verified == 1:  # 1 for true
            return jsonify({"error": "Email already verified"}), 400
        
        # Check code match and expiration
        if stored_code != verification_code:
            return jsonify({"error": "Invalid verification code"}), 401
        
        expires_at_dt = datetime.fromisoformat(expires_at)
        if datetime.utcnow() > expires_at_dt:
            return jsonify({"error": "Verification code has expired"}), 401
        
        # Update email_verified to true and clear verification code for security
        cursor.execute("UPDATE users SET email_verified = 1, verification_code = NULL, verification_code_expires_at = NULL WHERE user_id = ?", (user_id,))
        conn.commit()
        
        print(f"--- Email verified successfully for user {user_id} ---")
        return jsonify({"message": "Email verified successfully"}), 200
    except sqlite3.Error as e:
        print(f"--- Verify email failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to verify email due to database issue"}), 500
    except Exception as e:
        print(f"--- Verify email failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to verify email"}), 500
    finally:
        if conn:
            conn.close()
#get settings endpoint
@app.route('/api/get_settings', methods=['POST'])
@token_required  # Use JWT decorator for authentication
def get_user_settings(user_id):
    """Endpoint to retrieve user settings based on authenticated user ID."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT email, organization_name, email_alerts, low_credit_threshold, usage_reports
            FROM users WHERE user_id = ?
        ''', (user_id,))
        user_settings = cursor.fetchone()
        if not user_settings:
            return jsonify({"error": "User not found"}), 404

        # Unpack and return settings, including email
        settings = {
            "email": user_settings[0],  # New field added
            "organization_name": user_settings[1],
            "email_alerts": bool(user_settings[2]),  # Ensure BOOLEAN is handled as Python bool
            "low_credit_threshold": float(user_settings[3]),
            "usage_reports": user_settings[4]
        }
        print(f"--- Retrieved settings for user {user_id} including email ---")
        return jsonify({"settings": settings}), 200
    except sqlite3.Error as e:
        print(f"--- Get settings failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve settings due to database issue"}), 500
    except Exception as e:
        print(f"--- Get settings failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve settings"}), 500
    finally:
        if conn:
            conn.close()
# --- User Settings Update Endpoint ---
@app.route('/api/update_settings', methods=['POST'])
@token_required  # Use JWT decorator for authentication
def update_user_settings(user_id):
    """Endpoint to update user settings based on authenticated user ID."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    updates = data.get("settings", {})
    if not updates:
        return jsonify({"error": "No settings provided to update"}), 400

    # Validate and sanitize inputs
    allowed_keys = ["organization_name", "email_alerts", "low_credit_threshold", "usage_reports"]
    invalid_keys = [key for key in updates if key not in allowed_keys]
    if invalid_keys:
        return jsonify({"error": f"Invalid settings keys: {', '.join(invalid_keys)}"}), 400

    # Build update query dynamically
    set_clause = []
    values = []
    for key in allowed_keys:
        if key in updates:
            set_clause.append(f"{key} = ?")
            values.append(updates[key])

    if not set_clause:
        return jsonify({"error": "No valid settings to update"}), 400

    values.append(user_id)  # Add user_id for WHERE clause
    query = f"UPDATE users SET {', '.join(set_clause)} WHERE user_id = ?"

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "User not found"}), 404

        print(f"--- Updated settings for user {user_id} ---")
        return jsonify({"message": "Settings updated successfully"}), 200
    except sqlite3.IntegrityError as e:
        print(f"--- Update settings failed (IntegrityError): {e} ---")
        return jsonify({"error": "Update conflict, possibly due to email or username constraints"}), 409
    except sqlite3.Error as e:
        print(f"--- Update settings failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to update settings due to database issue"}), 500
    except Exception as e:
        print(f"--- Update settings failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to update settings"}), 500
    finally:
        if conn:
            conn.close()            
# --- API Key Generation Endpoint ---
@app.route('/api/generate_api_key', methods=['POST'])
@token_required  # Enforce JWT authentication; user_id is passed to the function
def generate_api_key(user_id):
    """Generate a new API key for the authenticated user."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # No need for user_id in the request body; it's derived from JWT
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Verify user exists (though token_required should ensure this)
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "User not found"}), 404

        # Generate and insert key
        api_key = f"QC_{str(uuid.uuid4()).replace('-', '')}"
        cursor.execute("INSERT INTO api_keys (api_key, user_id) VALUES (?, ?)", (api_key, user_id))
        conn.commit()
        conn.close()
        print(f"--- API key generated for user {user_id} ---")
        return jsonify({"message": "API key generated successfully", "api_key": api_key}), 201
    except sqlite3.Error as e:
        print(f"--- API key generation failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to generate API key due to database issue"}), 500
    except Exception as e:
        print(f"--- API key generation failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to generate API key"}), 500
# --- Endpoint to Revoke an API Key ---
@app.route('/api/revoke_api_key', methods=['POST'])
@token_required  # Apply JWT verification decorator to ensure user is authenticated
def revoke_api_key(user_id):
    print(f"--- Attempting to revoke API key for authenticated user ID: {user_id} ---")
    if not request.is_json:
        print("--- Request Error: Not JSON ---")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    api_key_to_revoke = data.get("api_key")
    if not api_key_to_revoke:
        print("--- Request Error: 'api_key' is required in the request body ---")
        return jsonify({"error": "'api_key' is required"}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Verify that the API key belongs to the user before revoking
        cursor.execute("SELECT api_key FROM api_keys WHERE user_id = ? AND api_key = ?", (user_id, api_key_to_revoke))
        result = cursor.fetchone()
        if not result:
            print(f"--- Revocation failed: API key '{api_key_to_revoke[:4]}...{api_key_to_revoke[-4:]}' not found or does not belong to user {user_id} ---")
            return jsonify({"error": "API key not found or unauthorized"}), 404

        # Delete the API key
        cursor.execute("DELETE FROM api_keys WHERE api_key = ?", (api_key_to_revoke,))
        conn.commit()
        if cursor.rowcount == 0:
            print(f"--- Revocation failed: No rows affected for API key '{api_key_to_revoke[:4]}...{api_key_to_revoke[-4:]}' ---")
            return jsonify({"error": "Failed to revoke API key"}), 500  # This is unlikely but handles edge cases

        print(f"--- API key '{api_key_to_revoke[:4]}...{api_key_to_revoke[-4:]}' revoked successfully for user {user_id} ---")
        return jsonify({"message": "API key revoked successfully"}), 200

    except sqlite3.Error as e:
        print(f"--- Revoke API key failed (Database Error): {e} ---")
        traceback.print_exc()  # Log full traceback for debugging
        return jsonify({"error": "Failed to revoke API key due to database issue"}), 500
    except Exception as e:
        print(f"--- Revoke API key failed (Exception): {e} ---")
        traceback.print_exc()  # Log full traceback for debugging
        return jsonify({"error": "Failed to revoke API key"}), 500
    finally:
        if conn:
            conn.close()
            print("--- Database connection closed ---")
# --- Endpoint to Retrieve API Keys for a User ---
@app.route('/api/get_api_keys', methods=['POST'])
@token_required  # Apply JWT verification decorator
def get_api_keys(user_id):
    print(f"--- Retrieving API keys for authenticated user ID: {user_id} ---")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # SQL query to join api_keys with api_logs to get last_used and calls_count
        query = """
            SELECT ak.api_key, ak.created_at, MAX(al.timestamp) AS last_used, COUNT(al.log_id) AS calls_count 
            FROM api_keys ak 
            LEFT JOIN api_logs al ON ak.api_key = al.api_key 
            WHERE ak.user_id = ? 
            GROUP BY ak.api_key, ak.created_at
        """
        print(f"--- Executing SQL query: {query} with user_id: {user_id} ---")  # Debug log for query
        
        cursor.execute(query, (user_id,))
        api_keys_rows = cursor.fetchall()
        
        # Log the raw results for debugging, including any NULL values
        print(f"--- Raw API keys data retrieved: {api_keys_rows} ---")
        
        # Format the results into a list of dictionaries
        api_keys = []
        for row in api_keys_rows:
            api_key_dict = {
                "api_key": row[0],
                "created_at": row[1],
                "last_used": row[2],  # Could be None if no logs
                "calls_count": row[3]  # Should be 0 if no logs
            }
            api_keys.append(api_key_dict)
        
        # Log the final response structure for debugging
        print(f"--- Final API keys response data: {api_keys} ---")
        
        print(f"--- Retrieved {len(api_keys)} API keys for user {user_id} with last_used and calls_count ---")
        
        return jsonify({
            "message": "API keys retrieved successfully",
            "user_id": user_id,
            "api_keys": api_keys
        }), 200
    
    except sqlite3.Error as e:
        print(f"--- Get API keys failed (Database Error): {e} ---")
        traceback.print_exc()  # Print full traceback for detailed debugging
        return jsonify({"error": "Failed to retrieve API keys due to database issue"}), 500
    except Exception as e:
        print(f"--- Get API keys failed (Exception): {e} ---")
        traceback.print_exc()  # Print full traceback for detailed debugging
        return jsonify({"error": "Failed to retrieve API keys"}), 500
    finally:
        if conn:
            conn.close()
            print("--- Database connection closed ---")
# --- Flask Endpoint for Quality Compute ---
@app.route('/api/generate', methods=['POST'])
def quality_compute_generate():
    print("\n--- Received request to /api/generate ---")
    if not request.is_json:
        print("--- Request Error: Not JSON ---")
        return jsonify({"error": "Request must be JSON"}), 400

    # --- Authentication ---
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        print("--- Auth Error: Missing or invalid Authorization header ---")
        return jsonify({"error": "Authorization header with Bearer token is required"}), 401
    user_api_key = auth_header.split(' ')[1]
    user_id = validate_api_key(user_api_key)
    if not user_id:
        return jsonify({"error": "Invalid or unauthorized API key"}), 401
    print(f"--- Authenticated User ID: {user_id} ---")

    # --- Request Parsing ---
    data = request.get_json()
    print(f"--- Full Request JSON for user {user_id}: {json.dumps(data, indent=2)} ---")
    model_requested = data.get("model")
    input_data = data.get("input")

    if not model_requested:
        return jsonify({"error": "'model' field is required"}), 400
    if input_data is None:
        return jsonify({"error": "'input' field is required"}), 400

    if not isinstance(input_data, (str, list)):
        return jsonify({"error": "'input' field must be a string or a list of message objects"}), 400
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                return jsonify({"error": "Invalid message format in 'input' list. Each item must be a dict with 'role' and 'content'."}), 400

    if data.get("stream", False):
        print("--- Request Error: Streaming not supported ---")
        return jsonify({"error": "Streaming is not supported with Best-of-N aggregation"}), 400

    include_candidates = data.get("include_candidates", False)
    judge_model_requested = data.get("judge_model", None)
    evaluator_model_used = judge_model_requested if judge_model_requested else model_requested  # Simplified for now

    # Extract base model and N value
    match = re.match(r'^(.+?)(-B(\d*))?$', model_requested)
    if not match:
        base_model = model_requested
        n_value = 1
        print(f"--- Warning: Treating model '{model_requested}' as single call (N=1) due to parsing failure. ---")
    else:
        base_model = match.group(1)
        n_suffix_digits = match.group(3)
        if n_suffix_digits and n_suffix_digits.isdigit():
            n_value = int(n_suffix_digits)
        else:
            n_value = DEFAULT_N_VALUE
        if n_value <= 0:
            return jsonify({"error": f"Invalid N value ({n_value}) in model name. Must be > 0."}), 400

    # Collect passthrough parameters
    simulator_keys = {"model", "input", "stream", "include_candidates", "judge_model"}
    passthrough_params = {k: v for k, v in data.items() if k not in simulator_keys}

    # --- Credit System: Cost Estimation and Pre-call Check ---
    estimated_cost = estimate_cost(base_model, input_data, n_value, passthrough_params)
    user_credits = get_user_credits(user_id)
    if user_credits < estimated_cost:
        send_email_notification(user_id, "insufficient_funds", estimated_cost, user_credits)
        return jsonify({"error": "Insufficient credits. Payment required.", "estimated_cost": estimated_cost}), 402

    # --- Execute Call(s) ---
    final_response_text = None
    all_candidate_responses = []
    aggregated_usage = {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0}
    start_time = datetime.now()

    if n_value == 1:
        print(f"--- Executing single call for {base_model}... ---")
        result_text, usage_info, error_info = call_model_api(base_model, input_data, **passthrough_params)
        if error_info:
            return jsonify({"error": error_info["error_detail"]}), error_info["status_code"]
        if result_text is not None:
            all_candidate_responses = [result_text]
        if usage_info:
            aggregated_usage['input_tokens'] = usage_info.get('input_tokens', 0)
            aggregated_usage['output_tokens'] = usage_info.get('output_tokens', 0)
            aggregated_usage['reasoning_tokens'] = usage_info.get('reasoning_tokens', 0)
            aggregated_usage['total_tokens'] = usage_info.get('total_tokens', aggregated_usage['input_tokens'] + aggregated_usage['output_tokens'])
        final_response_text = result_text
    else:  # Best-of-N
        print(f"--- Executing Best-of-{n_value} for {base_model}... ---")
        candidate_results = []  # Store tuples of (text, usage_dict, error_info)
        futures = []
        max_workers = min(n_value, MAX_WORKERS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"--- Submitting {n_value} parallel API calls to base model '{base_model}' (Max Workers: {max_workers}) ---")
            for i in range(n_value):
                futures.append(
                    executor.submit(call_model_api, base_model, input_data, **passthrough_params)
                )

            print("--- Waiting for Provider API calls to complete... ---")
            successful_calls = 0
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                call_num = i + 1
                try:
                    result_text, usage_info, error_info = future.result()  # Get result from completed future

                    if error_info:
                        print(f"--- Error in parallel call {call_num}/{n_value}: Status Code {error_info['status_code']}, Detail: {error_info['error_detail']} ---")
                        continue  # Skip this call if there was an error

                    # Process even if result_text is None (e.g. blocked prompt), as usage might be valid
                    if result_text is not None or usage_info is not None:
                        successful_calls += 1
                        if result_text is not None:
                            candidate_results.append((result_text, usage_info))  # Store only valid text results for selection
                            print(f"--- Received valid response text ({len(candidate_results)}/{n_value}) ---")
                        else:
                            print(f"--- Call {call_num}/{n_value} returned None text (e.g., blocked) but may have usage. ---")

                        # Aggregate usage safely, checking if usage_info is valid
                        if usage_info and isinstance(usage_info, dict):
                            input_tok = usage_info.get('input_tokens', 0)
                            output_tok = usage_info.get('output_tokens', 0)
                            reason_tok = usage_info.get('reasoning_tokens', 0)
                            total_tok = usage_info.get('total_tokens', input_tok + output_tok)  # Calculate if missing

                            aggregated_usage['input_tokens'] += input_tok
                            aggregated_usage['output_tokens'] += output_tok
                            aggregated_usage['reasoning_tokens'] += reason_tok
                            aggregated_usage['total_tokens'] += total_tok
                            print(f"--- Call {call_num} usage: {usage_info} -> Aggregated: {aggregated_usage} ---")
                        else:
                            # Log warning if text was okay but usage was bad, or if both were None
                            if result_text is not None:
                                print(f"--- Warning: Call {call_num} returned valid text but missing/invalid usage info. Usage contribution is zero. ---")
                            else:  # Both text and usage were None from this call
                                print(f"--- Warning: Call {call_num} failed entirely (no text or usage). ---")
                    else:
                        # Log that a specific call failed entirely (both text and usage are None)
                        print(f"--- Warning: A provider API call (Call {call_num}/{n_value}) to {base_model} failed completely (returned None, None). ---")

                except Exception as exc:
                    # Log exception from a specific future but continue
                    print(f"--- Error: Parallel call {call_num}/{n_value} to {base_model} generated an exception: {exc} ---")
                    traceback.print_exc()  # Log the full traceback for debugging

        # Process results after all futures completed or timed out
        all_candidate_responses = [res[0] for res in candidate_results if res[0] is not None]  # Extract non-None text results for judging
        print(f"--- Collected {len(all_candidate_responses)} valid candidate texts out of {n_value} attempts ({successful_calls} successful calls) for {base_model}. ---")

        if not all_candidate_responses:
            # Check if ANY calls succeeded (even if text was None) to differentiate total failure vs. all blocked
            if successful_calls > 0:
                print(f"--- Error: All successful provider calls ({successful_calls}/{n_value}) resulted in None/empty text (e.g., all blocked). Cannot select best. ---")
                error_detail = f"All {successful_calls} successful calls to model API ({base_model}) returned no usable text content (possibly due to content filters). No response selected."
                final_response_text = None  # Explicitly set to None
                evaluator_model_used = "N/A"  # No judge used
                # Return aggregated usage if available, but flag error
                return jsonify({"error": error_detail, "usage": aggregated_usage}), 400  # Use 400 for client-side issues like blocked prompts
            else:
                # All calls failed entirely
                print(f"--- Error: Failed to get any valid responses or usage from model API for Best-of-N ({base_model}). ---")
                error_detail = f"Failed to get any successful responses (0/{n_value}) from model API ({base_model}) for Best-of-{n_value}. Check service logs and backend API key configurations."
                return jsonify({"error": error_detail}), 500
        else:
            # Select best response using judge (only if more than one candidate)
            if len(all_candidate_responses) > 1:
                print("--- Initiating judge selection process ---")
                selected_text, _, _ = select_best_response(input_data, all_candidate_responses, evaluator_model_used)  # Pass the configurable judge model
                final_response_text = selected_text  # Judge function handles fallback
                if final_response_text is None:
                    # This case should be rare due to fallback, but handle it
                    print("--- Error: Judge failed to select a best response, and fallback failed (this shouldn't happen if candidates exist). ---")
                    return jsonify({"error": "Internal error during judge selection"}), 500
                else:
                    print(f"--- Judge selected best response using {evaluator_model_used} ---")
            elif len(all_candidate_responses) == 1:  # Only one candidate, use it directly
                final_response_text = all_candidate_responses[0]
                print("--- Only one valid candidate text returned, using it directly (no judge needed). ---")
                evaluator_model_used = "N/A"  # No judge used

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"--- Request processing completed in {duration.total_seconds():.2f} seconds ---")

    # --- Credit System: Actual Cost Deduction ---
    actual_cost = calculate_actual_cost(aggregated_usage, base_model)
    deduct_user_credits(user_id, actual_cost)

    # Check for low/negative balance and send notifications
    updated_credits = get_user_credits(user_id)
    if updated_credits < 0:
        send_email_notification(user_id, "negative_balance", actual_cost, updated_credits)
    elif updated_credits < 10.0:  # Example threshold; make configurable
        send_email_notification(user_id, "low_balance", actual_cost, updated_credits)

    # --- Logging ---
    # Ensure final_response_text is a string for logging, even if None initially
    final_response_text_for_log = str(final_response_text) if final_response_text is not None else ""
    print(f"--- Final Selected Response (Snippet): {final_response_text_for_log[:100]}... ---")
    print(f"--- Final Aggregated Usage: {aggregated_usage} ---")
    try:
        log_api_call(
            user_id=user_id,
            api_key=user_api_key,
            model_used=base_model,  # Log the base model used
            n_calls=n_value,
            input_data=input_data,
            usage_info=aggregated_usage,  # Log the aggregated usage
            candidate_responses=all_candidate_responses,  # Log only the valid text candidates used for selection
            final_response=final_response_text_for_log,  # Log the final selected text
            evaluator_model=evaluator_model_used
        )
    except Exception as log_err:
        print(f"--- CRITICAL ERROR during logging API call: {log_err} ---")
        # Don't fail the request, but log the error clearly
        traceback.print_exc()

    # --- Prepare Response ---
    response_payload = {
        "selected_text": final_response_text,  # Return None if it was None (e.g., all blocked)
        "candidates_count": len(all_candidate_responses),  # Count of valid text candidates
        "usage": aggregated_usage,  # Include usage in the response
        "evaluator": evaluator_model_used  # Indicate judge model used (or N/A)
    }
    if include_candidates:
        response_payload["candidate_responses"] = all_candidate_responses  # Return only valid text candidates

    print("--- Request successful (or partially successful if text is None), returning result. ---")
    return jsonify(response_payload)

# --- Endpoint for Analytics Data ---
@app.route('/api/analytics', methods=['POST'])
@token_required
def get_analytics_data(user_id):
    print(f"--- Retrieving analytics for authenticated user ID: {user_id} with filters ---")
    data = request.get_json()
    time_range = data.get("time_range", "7d")  # e.g., "24h", "7d", "30d"
    granularity = data.get("granularity", "daily")  # e.g., "hourly", "daily", "weekly", "monthly"
    group_by = data.get("group_by", "model")  # e.g., "model", "api_key", "token_type"

    # Validate inputs
    if time_range not in ["24h", "7d", "30d"]:
        return jsonify({"error": "Invalid time range. Use '24h', '7d', or '30d'."}), 400
    if granularity not in ["hourly", "daily", "weekly", "monthly"]:
        return jsonify({"error": "Invalid granularity. Use 'hourly', 'daily', 'weekly', or 'monthly'."}), 400
    if group_by not in ["model", "api_key", "token_type"]:
        return jsonify({"error": "Invalid group_by. Use 'model', 'api_key', or 'token_type'."}), 400

    # Calculate time delta and start time
    if time_range == "24h":
        delta = timedelta(hours=24)
    elif time_range == "7d":
        delta = timedelta(days=7)
    else:  # "30d"
        delta = timedelta(days=30)
    start_time = (datetime.utcnow() - delta).isoformat()

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        time_bucket_expr = get_time_bucket_expression(granularity)  # Get the SELECT expression with alias

        if group_by == "token_type":
            # Use UNION ALL to unpivot token types, group by time_bucket
            query = f"""
                SELECT granularity_param as granularity, time_bucket, 'input' as token_type, SUM(tokens_in) as value
                FROM (SELECT ?, {time_bucket_expr}, tokens_in FROM api_logs WHERE user_id = ? AND timestamp >= ?)
                GROUP BY time_bucket
                UNION ALL
                SELECT granularity_param as granularity, time_bucket, 'output' as token_type, SUM(tokens_out) as value
                FROM (SELECT ?, {time_bucket_expr}, tokens_out FROM api_logs WHERE user_id = ? AND timestamp >= ?)
                GROUP BY time_bucket
                UNION ALL
                SELECT granularity_param as granularity, time_bucket, 'reasoning' as token_type, SUM(reasoning_tokens) as value
                FROM (SELECT ?, {time_bucket_expr}, reasoning_tokens FROM api_logs WHERE user_id = ? AND timestamp >= ?)
                GROUP BY time_bucket
                ORDER BY time_bucket ASC, token_type ASC
            """
            # Pass granularity as a parameter for consistency in subqueries
            cursor.execute(query, (granularity, user_id, start_time, granularity, user_id, start_time, granularity, user_id, start_time))
            results = cursor.fetchall()
            analytics_data = [
                {
                    "time_bucket": row[1],
                    "token_type": row[2],
                    "value": row[3]  # Raw sum of tokens
                } for row in results
            ]
        elif group_by == "api_key":
            query = f"""
                SELECT {time_bucket_expr}, api_key, SUM(tokens_in) as input_tokens, SUM(tokens_out) as output_tokens, 
                       SUM(reasoning_tokens) as reasoning_tokens, COUNT(*) as call_count
                FROM api_logs WHERE user_id = ? AND timestamp >= ?
                GROUP BY time_bucket, COALESCE(api_key, 'unknown')  # Handle NULL api_key with 'unknown'
                ORDER BY time_bucket ASC
            """
            cursor.execute(query, (user_id, start_time))
            results = cursor.fetchall()
            analytics_data = [
                {
                    "time_bucket": row[0],
                    "api_key": row[1] or 'unknown',  # Handle NULL in Python
                    "input_tokens": row[2],
                    "output_tokens": row[3],
                    "reasoning_tokens": row[4],
                    "call_count": row[5]
                } for row in results
            ]
        else:  # group_by == "model"
            query = f"""
                SELECT {time_bucket_expr}, model_used, SUM(tokens_in) as input_tokens, SUM(tokens_out) as output_tokens, 
                       SUM(reasoning_tokens) as reasoning_tokens, COUNT(*) as call_count
                FROM api_logs WHERE user_id = ? AND timestamp >= ?
                GROUP BY time_bucket, model_used
                ORDER BY time_bucket ASC
            """
            cursor.execute(query, (user_id, start_time))
            results = cursor.fetchall()
            analytics_data = [
                {
                    "time_bucket": row[0],
                    "model": row[1],
                    "input_tokens": row[2],
                    "output_tokens": row[3],
                    "reasoning_tokens": row[4],
                    "call_count": row[5]
                } for row in results
            ]

        conn.close()
        print(f"--- Analytics retrieved for user {user_id} with granularity '{granularity}', group_by '{group_by}' over {time_range} (raw data only) ---")
        return jsonify({
            "analytics": analytics_data,
            "time_range": time_range,
            "granularity": granularity,
            "group_by": group_by
        }), 200
    except sqlite3.Error as e:
        print(f"--- Get analytics failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve analytics due to database issue"}), 500
    except Exception as e:
        print(f"--- Get analytics failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve analytics"}), 500
# --- Endpoint for Fetching Detailed Logs ---
@app.route('/api/logs', methods=['POST'])
@token_required  # Apply JWT verification
def get_api_logs(user_id):
    print(f"--- Retrieving API logs for authenticated user ID: {user_id} ---")
    data = request.get_json()
    page = data.get("page", 1)  # Page number, starting at 1
    per_page = data.get("per_page", 10)  # Items per page
    offset = (page - 1) * per_page

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT log_id, timestamp, model_used, n_calls, tokens_in, tokens_out, reasoning_tokens
            FROM api_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (user_id, per_page, offset))
        logs = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM api_logs WHERE user_id = ?", (user_id,))
        total_logs = cursor.fetchone()[0]
        conn.close()

        log_list = [
            {
                "log_id": row[0],
                "timestamp": row[1],
                "model_used": row[2],
                "n_calls": row[3],
                "tokens_in": row[4],
                "tokens_out": row[5],
                "reasoning_tokens": row[6]
            } for row in logs
        ]
        print(f"--- Retrieved {len(log_list)} API logs for user {user_id}, page {page} ---")
        return jsonify({
            "logs": log_list,
            "total_logs": total_logs,
            "page": page,
            "per_page": per_page
        }), 200
    except sqlite3.Error as e:
        print(f"--- Get API logs failed (Database Error): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve API logs due to database issue"}), 500
    except Exception as e:
        print(f"--- Get API logs failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve API logs"}), 500

# --- Secured Admin Endpoint for Editing User Details ---
@app.route('/admin/edit_user', methods=['POST'])
def admin_edit_user():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    admin_password_provided = data.get("admin_password")
    if not admin_password_provided:
        return jsonify({"error": "Admin password is required"}), 401

    if admin_password_provided != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 401

    user_id = data.get("user_id")
    updates = {}
    if "name" in data:
        updates["name"] = data["name"]
    if "email" in data:
        updates["email"] = data["email"]
    if "username" in data:
        updates["username"] = data["username"]
    if "password" in data:
        hashed_password = bcrypt.hashpw(data["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        updates["password"] = hashed_password
    if "credits" in data:
        try:
            credits = float(data["credits"])
            updates["credits"] = credits
        except ValueError:
            return jsonify({"error": "Credits must be a number"}), 400

    if not updates:
        return jsonify({"error": "No fields to update provided"}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Build the update query dynamically
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(user_id)
        query = f"UPDATE users SET {set_clause} WHERE user_id = ?"

        cursor.execute(query, values)
        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "User not found"}), 404

        print(f"--- User {user_id} updated successfully by admin ---")
        return jsonify({"message": "User updated successfully"}), 200
    except sqlite3.IntegrityError as e:
        print(f"--- Admin update failed (IntegrityError): {e} ---")
        if "users.email" in str(e):
            error_msg = "Email already exists"
        elif "users.username" in str(e):
            error_msg = "Username already exists"
        else:
            error_msg = "Update conflict"
        return jsonify({"error": error_msg}), 409
    except Exception as e:
        print(f"--- Admin update failed (Exception): {e} ---")
        traceback.print_exc()
        return jsonify({"error": "Failed to update user"}), 500
    finally:
        if conn:
            conn.close()
# --- Secured Admin Endpoint for Viewing Users ---
@app.route('/admin/get_users', methods=['POST'])
def admin_get_users():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    admin_password_provided = data.get("admin_password")
    if not admin_password_provided:
        return jsonify({"error": "Admin password is required"}), 401

    if admin_password_provided != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 401

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name, email, username, credits, created_at FROM users")  # Exclude password for security
        users = cursor.fetchall()
        conn.close()

        user_list = [
            {
                "user_id": row[0],
                "name": row[1],
                "email": row[2],
                "username": row[3],
                "credits": row[4],
                "created_at": row[5]
            } for row in users
        ]
        print(f"--- Admin retrieved user list successfully ---")
        return jsonify({"message": "Users retrieved successfully", "users": user_list}), 200
    except Exception as e:
        print(f"--- Admin get users failed (Exception): {e} ---")
        traceback.print_exc()
        if conn:
            conn.close()
        return jsonify({"error": "Failed to retrieve users"}), 500
# --- Run the Flask App ---
# if __name__ == '__main__':
#      print(f"--- Starting Quality Compute API Simulator ---")
#      print(f"--- Listening on: 0.0.0.0:5002 ---")
#      print(f"--- Database Path: {DB_PATH} ---")
#      print(f"--- Judge Model: {JUDGE_MODEL} ---")  # Fallback judge model
#      print(f"--- Default N for Best-of: {DEFAULT_N_VALUE} ---")
#      print(f"--- Max Workers for Parallel Calls: {MAX_WORKERS} ---")
#      print("--- Configured Providers & Status ---")
#      print(f"    - OpenAI Base URL: {MODEL_API_BASE_URL} {'(Key Found)' if OPENAI_API_KEY != 'dummy-openai-key' else '(Key MISSING!)'}")
#      print(f"    - xAI Base URL: {XAI_API_BASE_URL} {'(Key Found)' if XAI_API_KEY else '(Key MISSING!)'}")
#      print(f"    - Anthropic: Manually handled with requests; ensure ANTHROPIC_API_KEY is set.")
#      print(f"    - DeepSeek Base URL: {DEEPSEEK_API_BASE_URL} {'(Client Initialized)' if deepseek_client else '(Key MISSING or Init Failed!)'}")
#      print(f"    - Gemini: {'(API Configured)' if GEMINI_API_KEY else '(Key MISSING!)'}")
#      # Note: Use waitress or gunicorn for production instead of Flask's built-in server
#      # Use threaded=True for better handling of concurrent requests if using Flask dev server
#      app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)