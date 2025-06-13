#!/usr/bin/env python3
"""
Create .gitkeep files to preserve empty directory structure in Git
"""

import os
from pathlib import Path


def create_gitkeep_files():
    """Create .gitkeep files in directories that should be tracked but might be empty"""

    directories_to_keep = [
        "data/measurements",
        "data/performance_logs",
        "data/calibration_data",
        "images/samples",
        "images/calibration",
        "images/test_objects",
        "images/results",
        "hardware/vivado/projects",
        "hardware/vivado/ip_repo",
        "hardware/vivado/constraints",
        "hardware/vivado/scripts",
        "hardware/vhdl/src",
        "hardware/vhdl/testbenches",
        "hardware/vhdl/simulation",
        "hardware/overlay_files",
        "software/notebooks/development",
        "software/notebooks/testing",
        "software/notebooks/demos",
        "tests/test_data",
        "docs/hardware",
        "docs/software",
        "docs/user_manual",
        "docs/datasheets"
    ]

    created_count = 0
    existing_count = 0

    print("🔧 Creating .gitkeep files for directory structure preservation...")
    print("=" * 70)

    for directory in directories_to_keep:
        # Ensure directory exists
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep file
        gitkeep_path = dir_path / ".gitkeep"

        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"✅ Created: {gitkeep_path}")
            created_count += 1
        else:
            print(f"⏭️  Exists:  {gitkeep_path}")
            existing_count += 1

    print("=" * 70)
    print(f"🎉 Summary:")
    print(f"   Created: {created_count} new .gitkeep files")
    print(f"   Existing: {existing_count} files already present")
    print(f"   Total directories preserved: {len(directories_to_keep)}")
    print(f"\n✅ Directory structure will now be preserved in Git!")


if __name__ == "__main__":
    create_gitkeep_files()
