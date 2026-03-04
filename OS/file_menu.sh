#!/bin/bash

while true
do
    echo "------ File Operations Menu ------"
    echo "1. Create File"
    echo "2. Delete File"
    echo "3. Copy File"
    echo "4. List Files"
    echo "5. Exit"
    echo "Enter your choice:"
    read choice

    case $choice in
        1)
            echo "Enter file name to create:"
            read fname
            touch $fname
            echo "File created successfully."
            ;;
        2)
            echo "Enter file name to delete:"
            read fname
            rm -i $fname
            ;;
        3)
            echo "Enter source file:"
            read source
            echo "Enter destination file:"
            read dest
            cp $source $dest
            echo "File copied successfully."
            ;;
        4)
            echo "Files in current directory:"
            ls -l
            ;;
        5)
            echo "Exiting..."
            break
            ;;
        *)
            echo "Invalid choice. Try again."
            ;;
    esac

    echo ""
done
