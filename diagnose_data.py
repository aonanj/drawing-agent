#!/usr/bin/env python3
"""
Diagnostic script to check raw data directory structure and identify potential issues.
"""

import sys
from pathlib import Path
import zipfile

def diagnose_raw_data(raw_dir: Path):
    """Diagnose issues with raw data directory."""
    
    print("=" * 70)
    print("USPTO Data Directory Diagnostics")
    print("=" * 70)
    print()
    
    if not raw_dir.exists():
        print(f"❌ ERROR: Raw data directory not found: {raw_dir}")
        print(f"   Create it with: mkdir -p {raw_dir}")
        print("   Then add USPTO bulk download folders")
        return False
    
    print(f"✓ Raw data directory exists: {raw_dir}")
    print()
    
    # Find bulk folders
    bulk_folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    
    if not bulk_folders:
        print("❌ ERROR: No bulk download folders found in raw data directory")
        print()
        print("Expected structure:")
        print("  data/raw/")
        print("    └── redbook_2024_01/       (bulk download folder)")
        print("        ├── US12345678.zip     (individual patent zip)")
        print("        └── US12345679.zip")
        print()
        print("Download USPTO patent data from: https://bulkdata.uspto.gov/")
        return False
    
    print(f"✓ Found {len(bulk_folders)} bulk download folder(s):")
    for folder in bulk_folders:
        print(f"   - {folder.name}")
    print()
    
    # Analyze each bulk folder
    total_zips = 0
    total_valid = 0
    total_multiple_xml = 0
    total_no_xml = 0
    total_no_tiff = 0
    total_bad_zip = 0
    
    print("Analyzing patent zip files...")
    print("-" * 70)
    
    for bulk_folder in bulk_folders:
        zip_files = list(bulk_folder.glob("*.ZIP")) + list(bulk_folder.glob("*.zip"))
        total_zips += len(zip_files)
        
        print(f"\n{bulk_folder.name}: {len(zip_files)} zip file(s)")
        
        if not zip_files:
            print("  ⚠️  No zip files found in this folder")
            continue
        
        # Sample first few zips
        sample_size = min(5, len(zip_files))
        print(f"  Checking first {sample_size} files...")
        
        for zip_path in zip_files[:sample_size]:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    files = zf.namelist()
                    xml_files = [f for f in files if f.endswith(".xml") or f.endswith(".XML")]
                    tiff_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
                    
                    status = "✓"
                    notes = []
                    
                    if len(xml_files) == 0:
                        status = "❌"
                        notes.append("NO XML")
                        total_no_xml += 1
                    elif len(xml_files) > 1:
                        status = "⚠️"
                        notes.append(f"{len(xml_files)} XML files (will be skipped)")
                        total_multiple_xml += 1
                    
                    if len(tiff_files) == 0:
                        status = "⚠️"
                        notes.append("NO TIFF")
                        total_no_tiff += 1
                    
                    if status == "✓" and len(xml_files) == 1 and len(tiff_files) > 0:
                        total_valid += 1
                    
                    note_str = ", ".join(notes) if notes else "OK"
                    print(f"    {status} {zip_path.name}: {len(xml_files)} XML, {len(tiff_files)} TIFF - {note_str}")
                    
            except zipfile.BadZipFile:
                print(f"    ❌ {zip_path.name}: CORRUPTED ZIP FILE")
                total_bad_zip += 1
            except Exception as e:
                print(f"    ❌ {zip_path.name}: ERROR - {e}")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total zip files found:        {total_zips}")
    print(f"Valid patents (1 XML + TIFF): {total_valid}")
    print()
    
    if total_multiple_xml > 0:
        print(f"⚠️  Skipped (multiple XML):    {total_multiple_xml}")
        print("   (Patents with multiple XML files are skipped per requirements)")
    
    if total_no_xml > 0:
        print(f"❌ No XML files:               {total_no_xml}")
    
    if total_no_tiff > 0:
        print(f"⚠️  No TIFF files:             {total_no_tiff}")
        print("   (Patents without drawings will be skipped)")
    
    if total_bad_zip > 0:
        print(f"❌ Corrupted zip files:        {total_bad_zip}")
    
    print()
    
    if total_valid == 0:
        print("❌ NO VALID PATENTS FOUND!")
        print()
        print("Possible issues:")
        print("1. Zip files are corrupted or incomplete")
        print("2. Wrong directory structure (zips should be directly in bulk folders)")
        print("3. Downloaded wrong type of USPTO data (need patent grants with drawings)")
        print()
        print("Download correct data from: https://bulkdata.uspto.gov/")
        return False
    
    print(f"✓ Found {total_valid} valid patents that can be processed")
    print()
    print("Next steps:")
    print("  1. Test with single patent: python test_single_patent.py <path-to-zip>")
    print("  2. Process all: bash scripts/process_uspto.sh")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose USPTO raw data directory"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    success = diagnose_raw_data(args.raw_dir)
    sys.exit(0 if success else 1)