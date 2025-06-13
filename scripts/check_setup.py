import os


def check_project_structure():
    """Check if all required folders and files exist"""

    # Required folders
    required_folders = [
        "hardware",
        "hardware/vivado",
        "hardware/vhdl",
        "software",
        "software/python",
        "software/notebooks",
        "tests",
        "images",
        "data",
        "scripts",
        "config"
    ]

    # Required files
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "config/settings.py",
        "software/python/__init__.py",
        "tests/__init__.py"
    ]

    print("🔍 Checking project structure...")
    print("=" * 50)

    # Check folders
    missing_folders = []
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✅ Folder: {folder}")
        else:
            print(f"❌ Missing folder: {folder}")
            missing_folders.append(folder)

    # Check files
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ File: {file}")
        else:
            print(f"❌ Missing file: {file}")
            missing_files.append(file)

    print("=" * 50)

    if not missing_folders and not missing_files:
        print("🎉 SUCCESS! Project structure is complete!")
        return True
    else:
        print("❌ Some items are missing:")
        for item in missing_folders + missing_files:
            print(f"   - {item}")
        return False


if __name__ == "__main__":
    check_project_structure()
