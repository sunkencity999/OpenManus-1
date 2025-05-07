#!/usr/bin/env python3
import os
from pathlib import Path

def test_file_save():
    """
    Test if we can create and save files in the workspace directory.
    This will help us understand if there's an issue with file permissions
    or directory structure.
    """
    print("Testing file save functionality...")
    
    # Get the current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Define the workspace directory
    workspace_dir = os.path.join(cwd, "workspace")
    print(f"Workspace directory: {workspace_dir}")
    
    # Create the workspace directory if it doesn't exist
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"Created/confirmed workspace directory exists")
    
    # Create a test file
    test_file_path = os.path.join(workspace_dir, "test_file.txt")
    print(f"Attempting to create file at: {test_file_path}")
    
    try:
        # Write content to the file
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("This is a test file to verify file creation works.")
        print(f"Successfully created test file at: {test_file_path}")
        
        # Read back the content to verify
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"File content: {content}")
        
        # Check if the file exists
        if os.path.exists(test_file_path):
            print(f"File exists at: {test_file_path}")
        else:
            print(f"ERROR: File does not exist at: {test_file_path}")
            
        return True
    except Exception as e:
        print(f"ERROR: Failed to create/read file: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Run the test
    success = test_file_save()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
