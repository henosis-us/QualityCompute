import logging
import traceback
from flask import Blueprint, request, jsonify
import sqlite3
import bcrypt
import jwt
import os
import secrets # For generating secure tokens
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import send_email # Make sure this import works

load_dotenv()
auth_bp = Blueprint('auth', __name__)

DB_PATH = os.environ.get("DB_PATH", "quality_compute.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "your_jwt_secret_here")
# Define the base URL for your frontend application for password reset links
FRONTEND_BASE_URL = os.environ.get("REACT_APP_FRONTEND_URL", "https://qualitycompute.henosis.us")


# Configure logging for this blueprint
logger = logging.getLogger(__name__)
# Ensure logger is configured, e.g., in your main app file or here:
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@auth_bp.route('/api/login', methods=['POST'])
def login():
    # Log the receipt of the request with timestamp and headers
    logger.info(f"Received {request.method} request to /login at {datetime.now()}")
    logger.debug(f"Request Headers: {request.headers}")
    if request.method == 'OPTIONS':
        logger.debug("Handling preflight OPTIONS request for /login")
        return jsonify({}), 200
    if not request.is_json:
        logger.warning("Login request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    logger.info(f"Login attempt for email: {data.get('email')}, password provided: {'yes' if 'password' in data else 'no'}")
    email = data.get("email")
    password = data.get("password")
    if not all([email, password]):
        logger.warning("Missing fields in login request: email or password not provided")
        return jsonify({"error": "Email and password are required"}), 400
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logger.info(f"Executing database query for email: {email}")
        cursor.execute("SELECT user_id, name, email, username, password, email_verified FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            logger.warning(f"Login failed: No user found for email {email}")
            return jsonify({"error": "Invalid email or password"}), 401
        user_id, name, email_db, username, hashed_password, email_verified = user
        if email_verified == 0:
            logger.warning(f"Login denied: Email not verified for user {user_id}")
            return jsonify({"error": "Email not verified. Please verify your email before logging in."}), 403
        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            logger.warning(f"Login failed: Incorrect password for email {email}")
            return jsonify({"error": "Invalid email or password"}), 401
        token_payload = {
            "user_id": user_id,
            "email": email_db,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(token_payload, JWT_SECRET, algorithm="HS256")
        logger.info(f"Login successful for user {user_id}")
        return jsonify({
            "message": "Login successful",
            "user": {"user_id": user_id, "name": name, "email": email_db, "username": username},
            "token": token
        }), 200
    except sqlite3.Error as db_err:
        logger.error(f"Database error during login: {db_err}")
        traceback.print_exc()
        return jsonify({"error": "Database error during login"}), 500
    except jwt.InvalidTokenError as jwt_err:
        logger.error(f"JWT error during login: {jwt_err}")
        traceback.print_exc()
        return jsonify({"error": "JWT error during login"}), 500
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        traceback.print_exc()
        return jsonify({"error": "Login failed due to server error"}), 500
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after login attempt")

@auth_bp.route('/api/change_password', methods=['POST'])
def change_password():
    logger.info(f"Received {request.method} request to /change_password at {datetime.now()}")
    logger.debug(f"Request Headers: {request.headers}")
    
    # Token validation (should be done by a decorator ideally, but adding basic check here if not already)
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Change password request missing or invalid Authorization header")
        return jsonify({"error": "Authorization header with Bearer token is required"}), 401
    
    token = auth_header.split(' ')[1]
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"]) # Validate token
    except jwt.ExpiredSignatureError:
        logger.warning("Change password failed: Token has expired")
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        logger.warning("Change password failed: Invalid token")
        return jsonify({"error": "Invalid token"}), 401

    if not request.is_json:
        logger.warning("Change password request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    logger.info(f"Change password attempt for user_id: {data.get('user_id')}, new password provided: {'yes' if 'new_password' in data else 'no'}")
    
    user_id = data.get("user_id") # user_id from request body
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    
    if not all([user_id, current_password, new_password]):
        logger.warning("Missing fields in change password request")
        return jsonify({"error": "user_id, current_password, and new_password are required"}), 400

    if len(new_password) < 8: # Basic password policy
        logger.warning(f"New password for user {user_id} is too short.")
        return jsonify({"error": "New password must be at least 8 characters long."}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logger.info(f"Executing database query for user_id: {user_id}")
        cursor.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Change password failed: User not found for user_id {user_id}")
            return jsonify({"error": "User not found"}), 404
        
        hashed_password_db = result[0] # Renamed to avoid confusion
        if not bcrypt.checkpw(current_password.encode('utf-8'), hashed_password_db.encode('utf-8')):
            logger.warning(f"Change password failed: Incorrect current password for user_id {user_id}")
            return jsonify({"error": "Current password is incorrect"}), 401
        
        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("UPDATE users SET password = ? WHERE user_id = ?", (new_hashed_password, user_id))
        conn.commit()
        logger.info(f"Password changed successfully for user {user_id}")
        return jsonify({"message": "Password updated successfully"}), 200
    except sqlite3.Error as db_err:
        logger.error(f"Database error during password change: {db_err}")
        traceback.print_exc()
        return jsonify({"error": "Failed to change password due to database error"}), 500
    except Exception as e:
        logger.error(f"Unexpected error during password change: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to change password due to server error"}), 500
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed after change password attempt")


@auth_bp.route('/api/forgot_password', methods=['POST'])
def forgot_password():
    logger.info(f"Received {request.method} request to /api/forgot_password at {datetime.now()}")
    if not request.is_json:
        logger.warning("Forgot password request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    email = data.get("email")

    if not email:
        logger.warning("Email not provided in forgot password request")
        return jsonify({"error": "Email is required"}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            user_id, user_name = user
            reset_token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=30) # Token expires in 30 minutes

            cursor.execute(
                "UPDATE users SET reset_password_token = ?, reset_password_token_expires_at = ? WHERE user_id = ?",
                (reset_token, expires_at.isoformat(), user_id)
            )
            conn.commit()

            reset_url = f"{FRONTEND_BASE_URL}/reset-password/{reset_token}"
            email_subject = "Password Reset Request - Quality Compute"
            email_body = (
                f"Hello {user_name},\n\n"
                f"You requested a password reset for your Quality Compute account.\n"
                f"Please click the link below to set a new password:\n{reset_url}\n\n"
                f"This link will expire in 30 minutes.\n\n"
                f"If you did not request this, please ignore this email.\n\n"
                f"Best regards,\nThe Quality Compute Team"
            )
            send_email(email, email_subject, email_body)
            logger.info(f"Password reset email sent to {email} for user_id {user_id}")
        else:
            logger.info(f"Password reset requested for non-existent email: {email}")
        
        # Always return a generic message to prevent email enumeration
        return jsonify({"message": "If an account with that email address exists, a password reset link has been sent."}), 200

    except sqlite3.Error as db_err:
        logger.error(f"Database error during forgot password: {db_err}")
        traceback.print_exc()
        # Still return a generic message to the client for security
        return jsonify({"message": "If an account with that email address exists, a password reset link has been sent."}), 200
    except Exception as e:
        logger.error(f"Unexpected error during forgot password: {e}")
        traceback.print_exc()
        # Still return a generic message
        return jsonify({"message": "If an account with that email address exists, a password reset link has been sent."}), 200
    finally:
        if conn:
            conn.close()

@auth_bp.route('/api/reset_password', methods=['POST'])
def reset_password():
    logger.info(f"Received {request.method} request to /api/reset_password at {datetime.now()}")
    if not request.is_json:
        logger.warning("Reset password request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    reset_token = data.get("reset_token")
    new_password = data.get("new_password")

    if not all([reset_token, new_password]):
        logger.warning("Missing reset_token or new_password in reset password request")
        return jsonify({"error": "Reset token and new password are required"}), 400

    if len(new_password) < 8: # Basic password policy
        logger.warning(f"New password provided with token {reset_token[:8]}... is too short.")
        return jsonify({"error": "Password must be at least 8 characters long."}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, reset_password_token_expires_at FROM users WHERE reset_password_token = ?",
            (reset_token,)
        )
        user = cursor.fetchone()

        if not user:
            logger.warning(f"Invalid or non-existent reset token used: {reset_token[:8]}...")
            return jsonify({"error": "Invalid or expired reset token."}), 400

        user_id, expires_at_iso = user
        expires_at_dt = datetime.fromisoformat(expires_at_iso)

        if datetime.utcnow() > expires_at_dt:
            logger.warning(f"Expired reset token used for user_id {user_id}: {reset_token[:8]}...")
            # Invalidate the token even if it's expired
            cursor.execute(
                "UPDATE users SET reset_password_token = NULL, reset_password_token_expires_at = NULL WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            return jsonify({"error": "Reset token has expired."}), 401
        
        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        cursor.execute(
            "UPDATE users SET password = ?, reset_password_token = NULL, reset_password_token_expires_at = NULL WHERE user_id = ?",
            (new_hashed_password, user_id)
        )
        conn.commit()

        logger.info(f"Password reset successfully for user_id {user_id} using token {reset_token[:8]}...")
        return jsonify({"message": "Password has been reset successfully."}), 200

    except sqlite3.Error as db_err:
        logger.error(f"Database error during reset password: {db_err}")
        traceback.print_exc()
        return jsonify({"error": "Failed to reset password due to a server error."}), 500
    except ValueError: # Handles potential errors from datetime.fromisoformat
        logger.error(f"Invalid timestamp format for reset_password_token_expires_at with token {reset_token[:8]}...")
        traceback.print_exc()
        return jsonify({"error": "Invalid or expired reset token."}), 400
    except Exception as e:
        logger.error(f"Unexpected error during reset password: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to reset password due to a server error."}), 500
    finally:
        if conn:
            conn.close()
