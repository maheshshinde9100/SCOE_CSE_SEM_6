#!/bin/bash

# Set threshold percentage
THRESHOLD=80

# Get current disk usage percentage of root (/)
USAGE=$(df / | grep / | awk '{print $5}' | sed 's/%//g')

echo "Current Disk Usage: $USAGE%"

if [ "$USAGE" -ge "$THRESHOLD" ]; then
    echo "⚠ ALERT: Disk usage has exceeded ${THRESHOLD}%!"
    # Uncomment below line to send email alert (if mail configured)
    # echo "Disk usage is ${USAGE}%" | mail -s "Disk Alert" user@example.com
else
    echo "Disk usage is under control."
fi
