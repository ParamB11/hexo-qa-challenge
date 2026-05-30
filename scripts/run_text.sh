#! /bin/bash -l


# Define a path for the handshake file
INFO_FILE="logs/text_server_info.txt"

# Write the server's URL to the file so the client can find it
echo "http://$HOSTNAME:8001" > $INFO_FILE
echo "Server address written to $INFO_FILE"

# Start the server (ensure host="0.0.0.0" is in your python script!)
python src/text_server.py

# Clean up the file when the server job eventually ends or is killed
rm -f $INFO_FILE