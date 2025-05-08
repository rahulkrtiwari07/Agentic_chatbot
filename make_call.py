from twilio.rest import Client

# Twilio credentials from your Twilio Console
account_sid = 'AC37f415e8338b8710348e344942099312'
auth_token = '680be449acfc01ba27ef92a72bca7762'
client = Client(account_sid, auth_token)

# Initialize the Twilio client
try:
    call = client.calls.create(
    from_="+16468322701",
    to="+9779825216736",
    url="https://a8c6-140-82-42-230.ngrok-free.app/voice",)

    print(call.sid)

except Exception as e:
    print(f"An error occurred: {e}")
