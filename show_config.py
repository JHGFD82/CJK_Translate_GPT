#!/usr/bin/env python3
"""
Show current professor configuration from .env file
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def make_safe_filename(name: str) -> str:
    """Convert a professor name to a safe filename by replacing spaces and special chars with underscores."""
    # Replace spaces and special characters with underscores
    safe_name = re.sub(r'[^\w\-_\.]', '_', name)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name.lower()

def load_professor_config():
    """Load professor configuration from environment variables."""
    professors = {}
    
    # Look for all PROF_[ID]_NAME variables
    for key, value in os.environ.items():
        if key.startswith('PROF_') and key.endswith('_NAME'):
            # Extract the ID from PROF_[ID]_NAME
            prof_id = key[5:-5]  # Remove 'PROF_' and '_NAME'
            
            # Get the corresponding keys
            primary_key_var = f'PROF_{prof_id}_KEY'
            backup_key_var = f'PROF_{prof_id}_BACKUP_KEY'
            
            # Check if required keys exist
            if primary_key_var in os.environ:
                safe_name = make_safe_filename(value)
                professors[safe_name] = {
                    'name': value,
                    'primary_key': primary_key_var,
                    'backup_key': backup_key_var,
                    'id': prof_id,
                    'safe_name': safe_name
                }
    
    return professors

def show_professor_config():
    """Display current professor configuration."""
    professors = load_professor_config()
    
    if not professors:
        print("❌ No professors configured in .env file")
        print("Please add professor configuration in the format:")
        print("PROF_[ID]_NAME=Professor Name")
        print("PROF_[ID]_KEY=api_key")
        print("PROF_[ID]_BACKUP_KEY=backup_api_key")
        return
    
    print("📋 Current Professor Configuration:")
    print("=" * 50)
    
    for safe_name, prof_info in professors.items():
        display_name = prof_info['name']
        prof_id = prof_info['id']
        primary_key = prof_info['primary_key']
        backup_key = prof_info['backup_key']
        
        print(f"\n👤 {display_name}")
        print(f"   Safe Name: {safe_name}")
        print(f"   Environment ID: {prof_id}")
        print(f"   Primary Key: {primary_key}")
        print(f"   Backup Key: {backup_key}")
        
        # Check if keys are set
        primary_set = "✅" if os.environ.get(primary_key) else "❌"
        backup_set = "✅" if os.environ.get(backup_key) else "❌"
        
        print(f"   Primary Key Set: {primary_set}")
        print(f"   Backup Key Set: {backup_set}")
        
        # Show token usage file
        token_file = f"data/token_usage_{safe_name}.json"
        print(f"   Token Usage File: {token_file}")
    
    print("\n" + "=" * 50)
    print("💡 Usage Tips:")
    print("   • Use either the full name or safe name for CLI commands")
    print("   • Example: python main.py \"Professor Heller\" CE -i file.pdf")
    print("   • Example: python main.py professor_heller CE -i file.pdf")
    print("   • Run python main.py --help for more information")

if __name__ == "__main__":
    show_professor_config()
