"""
Configuration settings for EduGrade AI
Centralized configuration management
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "EduGrade AI"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database Configuration
    supabase_url: str = Field(default="", env="SUPABASE_URL")
    supabase_key: str = Field(default="", env="SUPABASE_KEY")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # File Storage Configuration
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./processed", env="PROCESSED_DIR")
    grades_dir: str = Field(default="./grades", env="GRADES_DIR")
    exports_dir: str = Field(default="./exports", env="EXPORTS_DIR")
    
    # API Keys
    google_gemini_api_key: Optional[str] = Field(default=None, env="GOOGLE_GEMINI_API_KEY")
    google_vision_api_key: Optional[str] = Field(default=None, env="GOOGLE_VISION_API_KEY")
    google_credentials_path: Optional[str] = Field(default=None, env="GOOGLE_CREDENTIALS_PATH")
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")
    
    # Model Configuration
    yolo_model_path: Optional[str] = Field(default=None, env="YOLO_MODEL_PATH")
    trocr_model_name: str = Field(default="microsoft/trocr-base-handwritten", env="TROCR_MODEL_NAME")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    max_image_size: int = Field(default=4096, env="MAX_IMAGE_SIZE")
    
    # Language Configuration
    supported_languages: List[str] = Field(default=["en", "hi", "ta"], env="SUPPORTED_LANGUAGES")
    
    # Grading Configuration
    default_rubric_path: Optional[str] = Field(default=None, env="DEFAULT_RUBRIC_PATH")
    partial_marking_enabled: bool = Field(default=True, env="PARTIAL_MARKING_ENABLED")
    feedback_length: str = Field(default="medium", env="FEEDBACK_LENGTH")
    
    # Security Configuration
    jwt_secret_key: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    hash_salt: Optional[str] = Field(default=None, env="HASH_SALT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Processing Configuration
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    job_timeout: int = Field(default=300, env="JOB_TIMEOUT")  # 5 minutes
    cleanup_old_files: bool = Field(default=True, env="CLEANUP_OLD_FILES")
    file_retention_days: int = Field(default=30, env="FILE_RETENTION_DAYS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.upload_dir,
            self.processed_dir,
            self.grades_dir,
            self.exports_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def database_config(self) -> dict:
        """Get database configuration"""
        return {
            "supabase_url": self.supabase_url,
            "supabase_key": self.supabase_key,
            "echo": self.database_echo
        }
    
    @property
    def file_config(self) -> dict:
        """Get file storage configuration"""
        return {
            "upload_dir": self.upload_dir,
            "processed_dir": self.processed_dir,
            "grades_dir": self.grades_dir,
            "exports_dir": self.exports_dir
        }
    
    @property
    def api_config(self) -> dict:
        """Get API configuration"""
        return {
            "google_gemini_api_key": self.google_gemini_api_key,
            "google_vision_api_key": self.google_vision_api_key,
            "google_credentials_path": self.google_credentials_path,
            "perplexity_api_key": self.perplexity_api_key
        }
    
    @property
    def model_config(self) -> dict:
        """Get model configuration"""
        return {
            "yolo_model_path": self.yolo_model_path,
            "trocr_model_name": self.trocr_model_name,
            "confidence_threshold": self.confidence_threshold,
            "max_image_size": self.max_image_size,
            "supported_languages": self.supported_languages
        }
    
    @property
    def grading_config(self) -> dict:
        """Get grading configuration"""
        return {
            "default_rubric_path": self.default_rubric_path,
            "partial_marking_enabled": self.partial_marking_enabled,
            "feedback_length": self.feedback_length
        }
    
    @property
    def security_config(self) -> dict:
        """Get security configuration"""
        return {
            "jwt_secret_key": self.jwt_secret_key,
            "hash_salt": self.hash_salt,
            "cors_origins": self.cors_origins
        }
    
    @property
    def processing_config(self) -> dict:
        """Get processing configuration"""
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "job_timeout": self.job_timeout,
            "cleanup_old_files": self.cleanup_old_files,
            "file_retention_days": self.file_retention_days
        }

# Global settings instance
settings = Settings()

# Validation functions
def validate_api_keys() -> dict:
    """Validate that required API keys are present"""
    missing_keys = []
    
    if not settings.google_gemini_api_key:
        missing_keys.append("GOOGLE_GEMINI_API_KEY")
    
    # Google Vision API key or credentials path
    if not settings.google_vision_api_key and not settings.google_credentials_path:
        missing_keys.append("GOOGLE_VISION_API_KEY or GOOGLE_CREDENTIALS_PATH")
    
    return {
        "valid": len(missing_keys) == 0,
        "missing_keys": missing_keys
    }

def validate_directories() -> dict:
    """Validate that required directories exist and are writable"""
    directories = [
        settings.upload_dir,
        settings.processed_dir,
        settings.grades_dir,
        settings.exports_dir
    ]
    
    issues = []
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            issues.append(f"Directory does not exist: {directory}")
        elif not path.is_dir():
            issues.append(f"Path is not a directory: {directory}")
        elif not os.access(path, os.W_OK):
            issues.append(f"Directory is not writable: {directory}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }

def validate_model_paths() -> dict:
    """Validate that model paths exist"""
    issues = []
    
    if settings.yolo_model_path and not Path(settings.yolo_model_path).exists():
        issues.append(f"YOLO model not found: {settings.yolo_model_path}")
    
    if settings.default_rubric_path and not Path(settings.default_rubric_path).exists():
        issues.append(f"Default rubric not found: {settings.default_rubric_path}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }

def get_validation_summary() -> dict:
    """Get overall validation summary"""
    api_validation = validate_api_keys()
    directory_validation = validate_directories()
    model_validation = validate_model_paths()
    
    return {
        "overall_valid": all([
            api_validation["valid"],
            directory_validation["valid"],
            model_validation["valid"]
        ]),
        "api_keys": api_validation,
        "directories": directory_validation,
        "model_paths": model_validation
    }

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    database_echo: bool = True

class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "WARNING"
    database_echo: bool = False
    cors_origins: List[str] = ["https://yourdomain.com"]

class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    supabase_url: str = "https://test-project.supabase.co"
    supabase_key: str = "test-key"
    log_level: str = "DEBUG"

# Factory function to get settings based on environment
def get_settings(env: str = "development") -> Settings:
    """Get settings based on environment"""
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return Settings()

if __name__ == "__main__":
    # Print current settings
    print("Current Settings:")
    print(f"App Name: {settings.app_name}")
    print(f"Debug: {settings.debug}")
    print(f"Supabase URL: {settings.supabase_url}")
    print(f"Upload Dir: {settings.upload_dir}")
    
    # Print validation summary
    validation = get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"Overall Valid: {validation['overall_valid']}")
    print(f"API Keys Valid: {validation['api_keys']['valid']}")
    print(f"Directories Valid: {validation['directories']['valid']}")
    print(f"Model Paths Valid: {validation['model_paths']['valid']}")
