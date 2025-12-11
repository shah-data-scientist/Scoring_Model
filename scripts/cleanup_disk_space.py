"""
Analyze and clean up disk space in the repository.

This script identifies and optionally removes:
- Python cache files (__pycache__, *.pyc)
- Duplicate MLflow backups
- Temporary files
- Large unnecessary files

Usage:
    # Analyze only (safe)
    poetry run python scripts/cleanup_disk_space.py --analyze

    # Clean up (removes files)
    poetry run python scripts/cleanup_disk_space.py --clean
"""

import os
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent


def get_dir_size(path):
    """Calculate directory size in bytes."""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total


def format_size(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def analyze_disk_usage():
    """Analyze disk usage and identify cleanup opportunities."""
    print("=" * 80)
    print("DISK SPACE ANALYSIS")
    print("=" * 80)
    print()

    # Total repository size
    total_size = get_dir_size(PROJECT_ROOT)
    print(f"Total repository size: {format_size(total_size)}")
    print()

    # Analyze by directory
    print("=" * 80)
    print("DIRECTORY SIZES")
    print("=" * 80)

    directories = {}
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name not in ['.git', '.venv', 'venv']:
            size = get_dir_size(item)
            directories[item.name] = size

    # Sort by size
    for name, size in sorted(directories.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {format_size(size):>10} - {name}")

    print()

    # Find Python cache
    print("=" * 80)
    print("PYTHON CACHE FILES")
    print("=" * 80)

    pycache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))
    pyc_files = list(PROJECT_ROOT.rglob('*.pyc'))

    pycache_size = sum(get_dir_size(d) for d in pycache_dirs)
    pyc_size = sum(f.stat().st_size for f in pyc_files if f.exists())

    total_cache_size = pycache_size + pyc_size

    print(f"  __pycache__ directories: {len(pycache_dirs)} ({format_size(pycache_size)})")
    print(f"  .pyc files: {len(pyc_files)} ({format_size(pyc_size)})")
    print(f"  Total cache size: {format_size(total_cache_size)}")
    print()

    # Find backup directories
    print("=" * 80)
    print("BACKUP DIRECTORIES")
    print("=" * 80)

    backups = {
        'mlruns_backup': PROJECT_ROOT / 'mlruns_backup',
        'mlruns_full_backup': PROJECT_ROOT / 'mlruns_full_backup',
    }

    total_backup_size = 0
    for name, path in backups.items():
        if path.exists():
            size = get_dir_size(path)
            total_backup_size += size
            print(f"  {format_size(size):>10} - {name}")

    print(f"  {format_size(total_backup_size):>10} - TOTAL BACKUPS")
    print()

    # Large files
    print("=" * 80)
    print("LARGE FILES (>10 MB)")
    print("=" * 80)

    large_files = []
    for f in PROJECT_ROOT.rglob('*'):
        if f.is_file():
            try:
                size = f.stat().st_size
                if size > 10 * 1024 * 1024:  # 10 MB
                    large_files.append((f, size))
            except:
                pass

    large_files.sort(key=lambda x: x[1], reverse=True)

    for file, size in large_files[:20]:
        rel_path = file.relative_to(PROJECT_ROOT)
        print(f"  {format_size(size):>10} - {rel_path}")

    print()

    # Summary
    print("=" * 80)
    print("CLEANUP OPPORTUNITIES")
    print("=" * 80)

    potential_savings = total_cache_size + total_backup_size

    print(f"\nPotential disk space savings:")
    print(f"  Python cache:    {format_size(total_cache_size)}")
    print(f"  Backup dirs:     {format_size(total_backup_size)}")
    print(f"  " + "-" * 30)
    print(f"  TOTAL:           {format_size(potential_savings)}")
    print()

    return {
        'pycache_dirs': pycache_dirs,
        'pyc_files': pyc_files,
        'backups': backups,
        'total_cache_size': total_cache_size,
        'total_backup_size': total_backup_size,
        'potential_savings': potential_savings
    }


def clean_cache(data):
    """Remove Python cache files."""
    print("\n[1/2] Cleaning Python cache...")

    removed_dirs = 0
    removed_files = 0

    # Remove __pycache__ directories
    for pycache_dir in data['pycache_dirs']:
        try:
            if pycache_dir.exists():
                shutil.rmtree(pycache_dir)
                removed_dirs += 1
        except Exception as e:
            print(f"  Error removing {pycache_dir}: {e}")

    # Remove .pyc files
    for pyc_file in data['pyc_files']:
        try:
            if pyc_file.exists():
                pyc_file.unlink()
                removed_files += 1
        except Exception as e:
            print(f"  Error removing {pyc_file}: {e}")

    print(f"  Removed {removed_dirs} __pycache__ directories")
    print(f"  Removed {removed_files} .pyc files")
    print(f"  Freed: {format_size(data['total_cache_size'])}")


def clean_backups(data):
    """Remove redundant backup directories."""
    print("\n[2/2] Cleaning backup directories...")

    # Keep only mlruns (current) and mlruns_full_backup
    # Remove mlruns_backup (older duplicate)

    to_remove = PROJECT_ROOT / 'mlruns_backup'

    if to_remove.exists():
        print(f"  Removing mlruns_backup (duplicate)...")
        try:
            shutil.rmtree(to_remove)
            size = get_dir_size(to_remove) if to_remove.exists() else data['total_backup_size'] / 2
            print(f"  Removed mlruns_backup")
            print(f"  Freed: ~{format_size(size)}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  mlruns_backup already removed")


def main():
    """Main cleanup function."""

    if '--clean' in sys.argv:
        print("\n⚠️  CLEANUP MODE - Files will be deleted!\n")
        response = input("Are you sure you want to clean up? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            return

        # Analyze first
        data = analyze_disk_usage()

        print("\n" + "=" * 80)
        print("CLEANING UP")
        print("=" * 80)

        # Clean
        clean_cache(data)
        clean_backups(data)

        print("\n" + "=" * 80)
        print("CLEANUP COMPLETE")
        print("=" * 80)
        print(f"\nFreed approximately: {format_size(data['potential_savings'])}")
        print("\nRun the script again with --analyze to see new disk usage.")

    else:
        # Analyze only (default)
        analyze_disk_usage()

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\nTo clean up identified files, run:")
        print("  poetry run python scripts/cleanup_disk_space.py --clean")
        print("\nThis will:")
        print("  1. Remove all Python cache files (__pycache__, *.pyc)")
        print("  2. Remove duplicate backup directory (mlruns_backup)")
        print("  3. Keep mlruns and mlruns_full_backup")


if __name__ == "__main__":
    main()
