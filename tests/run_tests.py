#!/usr/bin/env python3
"""
Test runner for EduGrade AI
Provides a convenient way to run all tests with different configurations
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    command = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend(["--cov=agents", "--cov=api", "--cov=dashboard", "--cov-report=html", "--cov-report=term"])
    
    return run_command(command, "Unit Tests")

def run_integration_tests(verbose=False):
    """Run integration tests"""
    command = ["python", "-m", "pytest", "tests/", "-m", "integration"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, "Integration Tests")

def run_slow_tests(verbose=False):
    """Run slow tests"""
    command = ["python", "-m", "pytest", "tests/", "-m", "slow"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, "Slow Tests")

def run_specific_test(test_path, verbose=False):
    """Run a specific test file"""
    command = ["python", "-m", "pytest", test_path]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, f"Specific Test: {test_path}")

def lint_code():
    """Run code linting"""
    commands = [
        (["python", "-m", "flake8", "agents/", "api/", "dashboard/", "tests/"], "Flake8 Linting"),
        (["python", "-m", "black", "--check", "agents/", "api/", "dashboard/", "tests/"], "Black Formatting Check"),
        (["python", "-m", "mypy", "agents/", "api/", "dashboard/"], "MyPy Type Checking")
    ]
    
    results = []
    for command, description in commands:
        results.append(run_command(command, description))
    
    return all(results)

def format_code():
    """Format code with black"""
    command = ["python", "-m", "black", "agents/", "api/", "dashboard/", "tests/"]
    return run_command(command, "Code Formatting")

def check_imports():
    """Check if all required packages are installed"""
    required_packages = [
        "fastapi",
        "streamlit",
        "gradio",
        "opencv-python",
        "ultralytics",
        "transformers",
        "openai",
        "pandas",
        "plotly",
        "pytest"
    ]
    
    print(f"\n{'='*60}")
    print("Checking Required Packages")
    print(f"{'='*60}")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed")
    return True

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="EduGrade AI Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--slow", action="store_true", help="Run slow tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", help="Run specific test file")
    parser.add_argument("--check-imports", action="store_true", help="Check if all packages are installed")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    print("🧪 EduGrade AI Test Runner")
    print(f"Project root: {project_root}")
    
    # Check imports first
    if args.check_imports or args.all:
        if not check_imports():
            print("\n❌ Package check failed. Please install missing packages.")
            return 1
    
    results = []
    
    # Run specific test
    if args.test:
        results.append(run_specific_test(args.test, args.verbose))
    
    # Run unit tests
    if args.unit or args.all:
        results.append(run_unit_tests(args.verbose, args.coverage))
    
    # Run integration tests
    if args.integration or args.all:
        results.append(run_integration_tests(args.verbose))
    
    # Run slow tests
    if args.slow or args.all:
        results.append(run_slow_tests(args.verbose))
    
    # Run linting
    if args.lint or args.all:
        results.append(lint_code())
    
    # Format code
    if args.format:
        results.append(format_code())
    
    # If no specific tests were requested, run unit tests by default
    if not any([args.unit, args.integration, args.slow, args.all, args.lint, args.format, args.test, args.check_imports]):
        results.append(run_unit_tests(args.verbose, args.coverage))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if results:
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1
    else:
        print("No tests were run.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
