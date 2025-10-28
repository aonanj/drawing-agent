"""
Test Single Patent Processing

This script helps you test the processing pipeline with a single patent
to verify everything works before processing large batches.
"""

import sys
from pathlib import Path
import json

from src.tuning.xml_parser import USPTOXMLParser
from src.tuning.image_processing import TIFFImageProcessor, validate_image_quality
import zipfile

from src.tuning import config as path_config

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_single_patent(zip_path: Path, output_dir: Path):
    """
    Test processing a single patent zip file.
    
    Args:
        zip_path: Path to a single patent zip file
        output_dir: Directory for test outputs
    """
    print(f"Testing patent: {zip_path.name}")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract and list contents
    print("\n1. Zip Contents:")
    print("-" * 60)
    with zipfile.ZipFile(zip_path, "r") as zf:
        files = zf.namelist()
        xml_files = [f for f in files if f.endswith(".xml") or f.endswith(".XML")]
        tiff_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        
        print(f"   Total files: {len(files)}")
        print(f"   XML files: {len(xml_files)}")
        print(f"   TIFF files: {len(tiff_files)}")
        
        if len(xml_files) != 1:
            print(f"   ⚠️  Expected 1 XML file, found {len(xml_files)}")
            if len(xml_files) > 1:
                print("   This patent will be skipped during batch processing")
                return
        
        # Extract files
        for f in xml_files + tiff_files:
            zf.extract(f, temp_dir)
            print(f"   Extracted: {f}")
    
    # Step 2: Parse XML
    print("\n2. XML Parsing:")
    print("-" * 60)
    xml_path = temp_dir / xml_files[0]
    parser = USPTOXMLParser(xml_path)
    patent_data = parser.parse()
    
    if patent_data is None:
        print("   ❌ Failed to parse XML")
        return
    
    print(f"   ✓ Patent ID: {patent_data.patent_id}")
    print(f"   ✓ Publication Kind: {patent_data.pub_kind}")
    print(f"   ✓ Family ID: {patent_data.family_id}")
    print(f"   ✓ CPC Codes: {', '.join(patent_data.cpc_codes[:5])}")
    print(f"   ✓ Number of Claims: {len(patent_data.claims)}")
    print(f"   ✓ Figure Descriptions: {len(patent_data.figure_descriptions)}")
    
    # Show figure descriptions
    if patent_data.figure_descriptions:
        print("\n   Figure Descriptions:")
        for fig_no, fig_desc in list(patent_data.figure_descriptions.items())[:3]:
            print(f"      {fig_no}: {fig_desc.description[:100]}...")
            if fig_desc.diagram_type:
                print(f"         Type: {fig_desc.diagram_type}")
    
    # Show first claim
    if patent_data.claims:
        claim = patent_data.claims[0]
        print(f"\n   First Claim (#{claim.claim_num}):")
        print(f"      Independent: {claim.is_independent}")
        print(f"      Components: {', '.join(claim.components[:5])}")
        if claim.relations:
            print(f"      Relations: {claim.relations[0]}")
    
    # Step 3: Process Images
    print("\n3. Image Processing:")
    print("-" * 60)
    
    if not tiff_files:
        print("   ⚠️  No TIFF files found")
        return
    
    processor = TIFFImageProcessor(target_size=1024)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    all_results = []
    for tiff_file in tiff_files:
        tiff_path = temp_dir / tiff_file
        print(f"\n   Processing: {tiff_file}")
        
        try:
            results = processor.process_tiff(
                tiff_path,
                images_dir,
                patent_data.patent_id
            )
            
            print(f"      Generated {len(results)} figure(s)")
            
            for target_path, control_path in results:
                # Validate
                is_valid = validate_image_quality(target_path)
                status = "✓" if is_valid else "⚠️"
                print(f"      {status} {target_path.name}")
                
                all_results.append((target_path, control_path, is_valid))
                
        except Exception as e:
            print(f"      ❌ Failed: {e}")
    
    # Step 4: Generate Prompts
    print("\n4. Prompt Generation:")
    print("-" * 60)
    
    for idx, (target_path, control_path, is_valid) in enumerate(all_results[:3]):
        print(f"\n   Sample {idx + 1}:")
        
        # Try to identify figure number
        figure_no = None
        if patent_data.figure_descriptions:
            figure_no = list(patent_data.figure_descriptions.keys())[0]
        
        # Generate prompt (simplified version)
        prompt_parts = [
            "Style: USPTO patent line art, monochrome, 300dpi, white background."
        ]
        
        if figure_no and figure_no in patent_data.figure_descriptions:
            fig_desc = patent_data.figure_descriptions[figure_no]
            prompt_parts.append(f"Figure: {figure_no}")
            if fig_desc.diagram_type:
                prompt_parts.append(f"Type: {fig_desc.diagram_type}")
            prompt_parts.append(f"Description: {fig_desc.description[:150]}")
        
        if patent_data.claims:
            claim = patent_data.claims[0]
            if claim.components:
                prompt_parts.append(f"Objects: {', '.join(claim.components[:5])}")
        
        prompt_parts.append(
            "Prohibitions: no shading, no color, no text outside labels"
        )
        
        prompt = " ".join(prompt_parts)
        print(f"      {prompt[:200]}...")
    
    # Step 5: Save Sample Metadata
    print("\n5. Saving Sample Metadata:")
    print("-" * 60)
    
    metadata = {
        "patent_id": patent_data.patent_id,
        "pub_kind": patent_data.pub_kind,
        "family_id": patent_data.family_id,
        "cpc_codes": patent_data.cpc_codes,
        "num_figures": len(all_results),
        "valid_figures": sum(1 for _, _, valid in all_results if valid),
        "figure_descriptions": {
            k: {
                "description": v.description,
                "diagram_type": v.diagram_type
            }
            for k, v in patent_data.figure_descriptions.items()
        },
        "claims_summary": {
            "total": len(patent_data.claims),
            "independent": sum(1 for c in patent_data.claims if c.is_independent),
            "first_claim_components": patent_data.claims[0].components if patent_data.claims else []
        },
        "abstract": patent_data.abstract[:500]
    }
    
    metadata_path = output_dir / "sample_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved to: {metadata_path}")
    
    # Step 6: Display Images
    print("\n6. Output Files:")
    print("-" * 60)
    print(f"   Test outputs saved to: {output_dir}")
    print(f"   Images: {images_dir}")
    print(f"   Metadata: {metadata_path}")
    
    if all_results:
        print("\n   View images:")
        for target_path, control_path, is_valid in all_results:
            print(f"      Target: {target_path}")
            print(f"      Control: {control_path}")
    
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("\nIf everything looks good, proceed with batch processing:")
    print(f"   GOOGLE_DRIVE_PATH=/path/to/colab/drawing-agent bash {path_config.SCRIPTS_DIR}/process_uspto.sh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test processing a single patent to validate the pipeline"
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to a single patent zip file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_output"),
        help="Output directory for test results (default: test_output)"
    )
    
    args = parser.parse_args()
    
    if not args.zip_path.exists():
        print(f"Error: Zip file not found: {args.zip_path}")
        sys.exit(1)
    
    test_single_patent(args.zip_path, args.output)