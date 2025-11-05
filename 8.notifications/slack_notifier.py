import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

def send_slack_notification(
    webhook_url: str,
    message: str,
    details: Optional[Dict] = None,
    emoji: str = ":warning:",
    include_timestamp: bool = True
) -> bool:
    """
    Send a notification to Slack via webhook.
    
    Args:
        webhook_url: Slack webhook URL
        message: Main message text
        details: Optional dictionary of details to include in the message
        emoji: Emoji to prefix the message with
        include_timestamp: Whether to include a timestamp in the message
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Format the message
        formatted_message = f"{emoji} {message}"
        
        if include_timestamp:
            formatted_message += f"\n*Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add details if provided
        if details:
            details_json = json.dumps(details, indent=2)
            formatted_message += f"\n*Details*:\n```{details_json}```"
        
        # Prepare payload
        payload = {
            "text": formatted_message
        }
        
        # Send to Slack
        response = requests.post(webhook_url, json=payload)
        
        # Check response
        if response.status_code == 200:
            return True
        else:
            print(f"Failed to send Slack notification: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending Slack notification: {str(e)}")
        return False

def send_battery_alert(
    webhook_url: str,
    voltage: float,
    temperature: float,
    additional_details: Optional[Dict] = None
) -> bool:
    """
    Send a battery alert notification to Slack.
    
    Args:
        webhook_url: Slack webhook URL
        voltage: Battery voltage
        temperature: Battery temperature
        additional_details: Optional additional details to include
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create alert message
    message = "BATTERY ALERT"
    
    # Create details dictionary
    details = {
        "voltage": voltage,
        "temperature": temperature
    }
    
    # Add any additional details
    if additional_details:
        details.update(additional_details)
    
    # Send the notification
    return send_slack_notification(
        webhook_url=webhook_url,
        message=message,
        details=details,
        emoji=":warning:",
        include_timestamp=True
    )