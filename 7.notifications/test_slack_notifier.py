import os
from slack_notifier import send_slack_notification, send_battery_alert

# Get your webhook URL from environment variables or replace with your actual URL
WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "YOUR_WEBHOOK_URL_HERE")

def test_basic_notification():
    """Test sending a basic notification"""
    print("Testing basic notification...")
    success = send_slack_notification(
        webhook_url=WEBHOOK_URL,
        message="HELLO WORLD - This is a test notification"
    )
    print(f"Notification sent successfully: {success}")
    return success

def test_detailed_notification():
    """Test sending a notification with details"""
    print("Testing detailed notification...")
    details = {
        "param1": 105,
        "param2": "valid",
        "raw_data": {"field1": 42, "field2": "error-prone"}
    }
    success = send_slack_notification(
        webhook_url=WEBHOOK_URL,
        message="Detailed test notification",
        details=details,
        emoji=":information_source:"
    )
    print(f"Notification sent successfully: {success}")
    return success

def test_battery_alert():
    """Test sending a battery alert"""
    print("Testing battery alert...")
    success = send_battery_alert(
        webhook_url=WEBHOOK_URL,
        voltage=3.5,
        temperature=48,
        additional_details={
            "battery_id": "BAT-12345",
            "location": "Warehouse A"
        }
    )
    print(f"Battery alert sent successfully: {success}")
    return success

if __name__ == "__main__":
    # Run tests
    test_basic_notification()
    test_detailed_notification()
    test_battery_alert()