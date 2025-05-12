import stripe
import os
from dotenv import load_dotenv
import sqlite3
import jwt
from datetime import datetime
from flask import Blueprint, request, jsonify
from functools import wraps
import logging
from utils import send_email_notification
import traceback

load_dotenv()

payments_bp = Blueprint('payments', __name__)
DB_PATH = os.environ.get("DB_PATH", "quality_compute.db")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "your_jwt_secret_here")

stripe.api_key = STRIPE_SECRET_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            user_id = payload.get('user_id')
            if not user_id:
                return jsonify({"error": "Invalid token: missing user_id"}), 401
            return f(user_id, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except Exception:
            return jsonify({"error": "Token verification failed"}), 401
    return decorated

# New endpoint for creating Stripe Checkout Session
@payments_bp.route('/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(user_id):
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    credits = data.get("credits")
    
    if not isinstance(credits, (int, float)) or credits <= 0:
        return jsonify({"error": "Invalid 'credits' field. Must be a positive number."}), 400
    
    try:
        # Get frontend base URL from environment variables
        frontend_base_url = os.environ.get("REACT_APP_FRONTEND_URL", "https://qualitycompute.henosis.us")
        success_url = f"{frontend_base_url}/payment-success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{frontend_base_url}/payment-cancelled"
        
        # Calculate amount: $1.10 per credit (including 10% margin), convert to cents
        amount_per_credit_cents = 110  # $1.10 in cents
        quantity = int(credits)  # Ensure quantity is an integer
        
        # Create Stripe Checkout Session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],  # Supports card payments; add more if needed
            line_items=[
                {
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': 'Compute Credits',
                        },
                        'unit_amount': amount_per_credit_cents,  # Unit amount in cents per credit
                    },
                    'quantity': quantity,  # Number of credits to purchase
                },
            ],
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                'user_id': user_id,
                'credits_purchased': credits  # Store credits for webhook processing
            }
        )
        
        # Return the session URL for redirection
        return jsonify({'url': session.url}), 200
    
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout session: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({"error": "Server error"}), 500

@payments_bp.route('/create_payment_intent', methods=['POST'])
def create_payment_intent():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    amount = data.get("amount")
    currency = data.get("currency", "usd")
    user_id = data.get("user_id")

    if not all([amount, user_id]):
        return jsonify({"error": "amount and user_id are required"}), 400
    if amount <= 0:
        return jsonify({"error": "Amount must be greater than 0"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "User not found"}), 404
        conn.close()

        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            metadata={"user_id": user_id}
        )
        return jsonify({
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id
        }), 200
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Payment intent creation failed: {str(e)}")
        return jsonify({"error": "Server error"}), 500

@payments_bp.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        logger.error("Invalid webhook payload")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        logger.error("Webhook signature verification failed")
        return jsonify({"error": "Signature verification failed"}), 400

    if event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']
        payment_intent_id = payment_intent.id
        user_id = payment_intent.metadata.get('user_id')
        credits_to_add_str = payment_intent.metadata.get('credits_to_add', '0')

        if not user_id:
            logger.error("No user_id in metadata")
            return jsonify({"error": "No user_id in metadata"}), 400

        try:
            credits_to_add = float(credits_to_add_str)
        except ValueError:
            logger.error(f"Invalid credits_to_add: {credits_to_add_str}")
            credits_to_add = 0.0

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            conn.execute("BEGIN")  # Start transaction explicitly for atomicity

            try:
                # Step 1: Attempt to record the payment first to ensure idempotency
                cursor.execute("INSERT INTO payment_records (payment_intent_id, user_id, credits_added) VALUES (?, ?, ?)",
                               (payment_intent_id, user_id, credits_to_add))
                
                # If insert succeeds, proceed to add credits
                cursor.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (credits_to_add, user_id))
                logger.info(f"Credits added: {credits_to_add} to user {user_id}")

                # Fetch updated credits for email notification
                cursor.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
                new_credits_result = cursor.fetchone()
                if new_credits_result is None:
                    raise Exception(f"User {user_id} not found after credit update attempt.")
                new_credits = new_credits_result[0]

                # Commit the transaction after both operations
                conn.commit()

                # Send thank you email notification outside the transaction
                try:
                    send_email_notification(user_id, "purchase_confirmation", credits_added=credits_to_add, credits=new_credits)
                    logger.info(f"Purchase confirmation email sent to user {user_id}")
                except Exception as email_err:
                    logger.error(f"Failed to send purchase confirmation email after successful credit update: {str(email_err)}")
                    # Do not fail the webhook; email failure is non-critical for Stripe's perspective

                return jsonify({"message": "Credits added successfully"}), 200

            except sqlite3.IntegrityError as ie:
                # Handle idempotency: UNIQUE constraint violation means already processed
                conn.rollback()  # Rollback any uncommitted changes
                if 'UNIQUE constraint failed: payment_records.payment_intent_id' in str(ie):
                    logger.info(f"Payment intent {payment_intent_id} already processed (detected by duplicate insert attempt), skipping.")
                    return jsonify({"message": "Payment already processed"}), 200
                else:
                    # Other integrity errors, log and reraise
                    logger.error(f"Unexpected IntegrityError during payment processing: {str(ie)}")
                    raise

        except Exception as e:
            # General error handling
            if conn:
                conn.rollback()  # Ensure rollback on any error
            logger.error(f"Credit update failed: {str(e)}")
            logger.error(traceback.format_exc())  # Log full traceback for debugging
            return jsonify({"error": "Failed to process payment webhook"}), 500

        finally:
            if conn:
                conn.close()

    else:
        return jsonify({"message": "Event not handled"}), 200

@payments_bp.route('/create_payment_for_credits', methods=['POST'])
@token_required
def create_payment_for_credits(user_id):
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    credits_requested = data.get("credits")

    if not isinstance(credits_requested, (int, float)) or credits_requested <= 0:
        return jsonify({"error": "Invalid 'credits' field. Must be a positive number."}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            return jsonify({"error": "User not found"}), 404
        conn.close()

        amount_cents = int(credits_requested * 110)  # Apply 10% margin
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            metadata={
                "user_id": user_id,
                "credits_to_add": credits_requested
            }
        )
        return jsonify({
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "amount": amount_cents / 100.0,
            "credits_requested": credits_requested
        }), 200
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Payment creation failed: {str(e)}")
        return jsonify({"error": "Server error"}), 500