#!/usr/bin/env python3
"""
Setup script for Google Gemini API
Helps users configure their Gemini API key for EduGrade AI
"""

import os
import sys
from pathlib import Path

def get_gemini_api_key():
    """Get Gemini API key from user input"""
    print("🔑 Google Gemini API Setup")
    print("=" * 50)
    print()
    print("To use EduGrade AI, you need a Google Gemini API key.")
    print("Follow these steps to get your API key:")
    print()
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    print()
    
    api_key = input("Enter your Google Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return None
    
    if not api_key.startswith('AI'):
        print("⚠️  Warning: Gemini API keys usually start with 'AI'. Please verify your key.")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return None
    
    return api_key

def update_env_file(api_key):
    """Update .env file with the API key"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    # Create .env from example if it doesn't exist
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print(f"✅ Created .env file from env.example")
    
    # Read current .env content
    if env_file.exists():
        content = env_file.read_text()
    else:
        content = ""
    
    # Update or add the API key
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith('GOOGLE_GEMINI_API_KEY='):
            lines[i] = f'GOOGLE_GEMINI_API_KEY={api_key}'
            updated = True
            break
    
    if not updated:
        lines.append(f'GOOGLE_GEMINI_API_KEY={api_key}')
    
    # Write back to file
    env_file.write_text('\n'.join(lines))
    print(f"✅ Updated .env file with Gemini API key")

def test_gemini_connection(api_key):
    """Test the Gemini API connection"""
    try:
        import google.generativeai as genai
        
        print("\n🧪 Testing Gemini API connection...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test
        response = model.generate_content("Hello, this is a test.")
        
        if response.text:
            print("✅ Gemini API connection successful!")
            print(f"Test response: {response.text[:100]}...")
            return True
        else:
            print("❌ Gemini API connection failed - no response")
            return False
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 EduGrade AI - Gemini API Setup")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the EduGrade AI project root directory")
        sys.exit(1)
    
    # Get API key from user
    api_key = get_gemini_api_key()
    if not api_key:
        sys.exit(1)
    
    # Update .env file
    update_env_file(api_key)
    
    # Test connection
    if test_gemini_connection(api_key):
        print("\n🎉 Setup complete! You can now run EduGrade AI.")
        print("\nNext steps:")
        print("1. Run: make install-dev")
        print("2. Run: make quick-start")
        print("3. Open your browser to http://localhost:8501")
    else:
        print("\n⚠️  Setup completed but API test failed.")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()
