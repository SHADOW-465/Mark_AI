#!/usr/bin/env python3
"""
Security Configuration Validator for EduGrade AI
Validates JWT and hash salt configuration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import settings
except ImportError as e:
    print(f"❌ Error importing settings: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def validate_jwt_secret():
    """Validate JWT secret key configuration"""
    print("🔐 Validating JWT Secret Key...")
    
    if not settings.jwt_secret_key:
        print("❌ JWT_SECRET_KEY is not set")
        return False
    
    jwt_key = settings.jwt_secret_key
    
    # Check length
    if len(jwt_key) < 32:
        print(f"⚠️  Warning: JWT secret key is only {len(jwt_key)} characters")
        print("   Recommendation: Use at least 32 characters for security")
    else:
        print(f"✅ JWT secret key length: {len(jwt_key)} characters")
    
    # Check character diversity
    unique_chars = len(set(jwt_key))
    if unique_chars < 10:
        print(f"⚠️  Warning: JWT secret key has only {unique_chars} unique characters")
        print("   Recommendation: Use more diverse characters")
    else:
        print(f"✅ JWT secret key diversity: {unique_chars} unique characters")
    
    return True

def validate_hash_salt():
    """Validate hash salt configuration"""
    print("\n🧂 Validating Hash Salt...")
    
    if not settings.hash_salt:
        print("❌ HASH_SALT is not set")
        return False
    
    hash_salt = settings.hash_salt
    
    # Check length
    if len(hash_salt) < 16:
        print(f"⚠️  Warning: Hash salt is only {len(hash_salt)} characters")
        print("   Recommendation: Use at least 16 characters for security")
    else:
        print(f"✅ Hash salt length: {len(hash_salt)} characters")
    
    # Check character diversity
    unique_chars = len(set(hash_salt))
    if unique_chars < 8:
        print(f"⚠️  Warning: Hash salt has only {unique_chars} unique characters")
        print("   Recommendation: Use more diverse characters")
    else:
        print(f"✅ Hash salt diversity: {unique_chars} unique characters")
    
    return True

def test_security_functions():
    """Test security functions with current configuration"""
    print("\n🧪 Testing Security Functions...")
    
    try:
        import hashlib
        
        # Test hash salt functionality
        test_data = "test_security_data"
        salt = settings.hash_salt
        hash_result = hashlib.sha256((test_data + salt).encode()).hexdigest()
        print(f"✅ Hash salt test: {hash_result[:16]}...")
        
    except Exception as e:
        print(f"❌ Hash salt test failed: {e}")
        return False
    
    try:
        import jwt
        from datetime import datetime, timedelta
        
        # Test JWT functionality
        secret = settings.jwt_secret_key
        payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, secret, algorithm='HS256')
        print(f"✅ JWT test: {token[:50]}...")
        
        # Test JWT decoding
        decoded = jwt.decode(token, secret, algorithms=['HS256'])
        print(f"✅ JWT decode test: user_id={decoded['user_id']}")
        
    except ImportError:
        print("⚠️  JWT library not installed. Install with: pip install PyJWT")
        return False
    except Exception as e:
        print(f"❌ JWT test failed: {e}")
        return False
    
    return True

def check_environment_file():
    """Check if .env file exists and is readable"""
    print("\n📁 Checking Environment Configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Run: python scripts/generate_security_keys.py")
        return False
    
    print("✅ .env file found")
    
    # Check if file is readable
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            if 'JWT_SECRET_KEY' in content and 'HASH_SALT' in content:
                print("✅ Security keys found in .env file")
            else:
                print("⚠️  Security keys not found in .env file")
                return False
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False
    
    return True

def main():
    """Main validation function"""
    print("🔍 EduGrade AI Security Configuration Validator")
    print("=" * 50)
    
    # Check environment file
    env_ok = check_environment_file()
    
    # Validate JWT secret
    jwt_ok = validate_jwt_secret()
    
    # Validate hash salt
    salt_ok = validate_hash_salt()
    
    # Test security functions
    functions_ok = test_security_functions()
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 20)
    
    all_ok = env_ok and jwt_ok and salt_ok and functions_ok
    
    if all_ok:
        print("✅ All security configurations are valid!")
        print("\n🚀 Your EduGrade AI application is ready to run securely.")
    else:
        print("❌ Some security configurations need attention.")
        print("\n🔧 To fix issues:")
        print("1. Run: python scripts/generate_security_keys.py")
        print("2. Update your .env file with the generated keys")
        print("3. Re-run this validator")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

