import os
from dotenv import load_dotenv
import sqlite3
import smtplib
from email.mime.text import MIMEText

# Load environment variables to ensure they are available
load_dotenv()

# Define DB_PATH here for consistency, or it can be accessed via os.environ
DB_PATH = os.environ.get("DB_PATH", "quality_compute.db")

def send_email(to_email, subject, body):
    """
    Send an email using Google Workspace SMTP relay.
    """
    EMAIL_USER = os.environ.get("EMAIL_USER", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")  # Optional; can be removed if using IP-based auth

    if not EMAIL_USER:
        print("--- Error: EMAIL_USER not set in environment variables ---")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = to_email

    try:
        server = smtplib.SMTP('smtp-relay.gmail.com', 587)  # Use port 587 for TLS
        server.starttls()  # Enable TLS; remove if not using TLS
        # If using SMTP authentication, uncomment the line below
        # server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
        print(f"--- Email sent to {to_email} via Google Workspace SMTP relay ---")
    except Exception as e:
        print(f"--- Error sending email via SMTP relay: {e} ---")

def get_email_template(notification_type, cost, credits, credits_added=0.0):
    """
    Return email subject and body based on notification type.
    Args:
        notification_type (str): Type of notification.
        cost (float): Cost for context.
        credits (float): Current credits.
        credits_added (float): Credits added (default 0.0).
    """
    if notification_type == "insufficient_funds":
        subject = "Insufficient Credits for API Call"
        body = f"Your credit balance is too low to complete the request. Estimated cost: ${cost:.2f}. Current credits: ${credits:.2f}. Please add credits via the payment page."
    elif notification_type == "low_balance":
        subject = "Low Credit Balance Alert"
        body = f"Your credit balance is low (${credits:.2f}). Consider adding more credits to continue using the service."
    elif notification_type == "negative_balance":
        subject = "Negative Credit Balance"
        body = f"Your account has a negative balance (-${abs(credits):.2f}) after a recent API call costing ${cost:.2f}. Please add credits immediately."
    elif notification_type == "purchase_confirmation":
        subject = "Thank you for your purchase!"
        body = f"Thank you for purchasing {credits_added:.2f} credits. Your new balance is ${credits:.2f}."
    else:
        subject = "Notification from Quality Compute"
        body = "An issue occurred with your account. Please contact support."
    return subject, body

def send_email_notification(user_id, notification_type, cost=0.0, credits=0.0, credits_added=0.0):
    """
    Send an email notification based on the type.
    Args:
        user_id (str): The user's ID.
        notification_type (str): Type of notification.
        cost (float): Optional cost for context.
        credits (float): Optional current credits.
        credits_added (float): Optional credits added.
    """
    # Fetch user email from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE user_id = ?", (user_id,))
    user_email_result = cursor.fetchone()
    conn.close()
    
    if not user_email_result:
        print(f"--- Error: User {user_id} not found for email notification ---")
        return

    user_email = user_email_result[0]
    email_subject, email_body = get_email_template(notification_type, cost, credits, credits_added)
    send_email(user_email, email_subject, email_body)