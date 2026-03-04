#!/bin/bash

echo "Parent Process ID: $$"

echo "1. Display Running Processes (ps)"
ps -f

echo ""
echo "2. Creating Child Process (Simulating fork)"

# Fork simulation
(
    echo "Child Process Started"
    echo "Child PID: $$"
    sleep 5
    echo "Child Process Completed"
) &

CHILD_PID=$!

echo "Child Process ID: $CHILD_PID"

echo ""
echo "3. Parent Waiting for Child (Simulating wait/join)"
wait $CHILD_PID

echo "Child process has finished execution."

echo ""
echo "4. Demonstrating exec (Replacing current process)"

echo "Executing 'ls -l' using exec..."
exec ls -l

# Any code below exec will NOT execute
echo "This line will not execute"
