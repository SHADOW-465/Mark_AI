#!/usr/bin/env python3
"""
Security Key Generator for EduGrade AI
Generates secure JWT secret keys and hash salts
"""

import secrets
import string
import argparse
import sys
from pathlib import Path

def generate_jwt_secret(length=64):
    """Generate a secure JWT secret key"""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

def generate_hash_salt(length=32):
    """Generate a secure hash salt"""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))

def generate_env_file(jwt_secret, hash_salt, output_file=".env"):
    """Generate a .env file with security keys"""
    env_content = f"""# Security Configuration
JWT_SECRET_KEY={jwt_secret}
HASH_SALT={hash_salt}

# Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# API Keys
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
GOOGLE_VISION_API_KEY=your_vision_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key

# File Storage
UPLOAD_DIR=./uploads
PROCESSED_DIR=./processed
GRADES_DIR=./grades
EXPORTS_DIR=./exports

# Model Settings
YOLO_MODEL_PATH=./models/yolo_answer_sheet.pt
TROCR_MODEL_NAME=microsoft/trocr-base-handwritten
MAX_IMAGE_SIZE=4096
CONFIDENCE_THRESHOLD=0.5

# Grading Settings
DEFAULT_RUBRIC_PATH=./config/default_rubric.json
PARTIAL_MARKING_ENABLED=true
FEEDBACK_LENGTH=medium

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000", "https://yourdomain.com"]
"""
    
    with open(output_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Generated {output_file} with security keys")

def main():
    parser = argparse.ArgumentParser(description='Generate security keys for EduGrade AI')
    parser.add_argument('--jwt-length', type=int, default=64, help='JWT secret key length (default: 64)')
    parser.add_argument('--salt-length', type=int, default=32, help='Hash salt length (default: 32)')
    parser.add_argument('--output', type=str, default='.env', help='Output file (default: .env)')
    parser.add_argument('--keys-only', action='store_true', help='Print only the keys, not the full .env file')
    
    args = parser.parse_args()
    
    # Generate keys
    jwt_secret = generate_jwt_secret(args.jwt_length)
    hash_salt = generate_hash_salt(args.salt_length)
    
    if args.keys_only:
        print(f"JWT_SECRET_KEY={jwt_secret}")
        print(f"HASH_SALT={hash_salt}")
    else:
        # Check if .env already exists
        if Path(args.output).exists():
            response = input(f"{args.output} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                sys.exit(1)
        
        generate_env_file(jwt_secret, hash_salt, args.output)
        
        print(f"\n🔐 Security Keys Generated:")
        print(f"JWT Secret Key: {jwt_secret[:16]}... (length: {len(jwt_secret)})")
        print(f"Hash Salt: {hash_salt[:16]}... (length: {len(hash_salt)})")
        print(f"\n📝 Next steps:")
        print(f"1. Update your Supabase URL and key in {args.output}")
        print(f"2. Add your API keys to {args.output}")
        print(f"3. Run the application with: python -m uvicorn api.main:app --reload")

if __name__ == "__main__":
    main()

