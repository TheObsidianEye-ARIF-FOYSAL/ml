import subprocess
import sys

def git_auto_push():
    """
    Automates git add, commit, and push operations.
    Takes commit message as input from the user.
    """
    try:
        # Get commit message from user
        commit_message = input("Enter commit message: ")
        
        if not commit_message.strip():
            print("Error: Commit message cannot be empty!")
            sys.exit(1)
        
        # Git add
        print("\n--- Running: git add . ---")
        result = subprocess.run(["git", "add", "."], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Error in git add: {result.stderr}")
            sys.exit(1)
        print("✓ Files staged successfully")
        
        # Git commit
        print(f"\n--- Running: git commit -m \"{commit_message}\" ---")
        result = subprocess.run(["git", "commit", "-m", commit_message], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Error in git commit: {result.stderr}")
            sys.exit(1)
        print(f"✓ Committed successfully\n{result.stdout}")
        
        # Git push
        print("\n--- Running: git push origin main ---")
        result = subprocess.run(["git", "push", "origin", "main"], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Error in git push: {result.stderr}")
            sys.exit(1)
        print(f"✓ Pushed successfully\n{result.stdout}")
        
        print("\n🎉 All operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    git_auto_push()
