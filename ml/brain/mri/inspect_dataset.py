"""
BraTS Dataset-Wide Inspector - Summary Report
==============================================

Analyzes ENTIRE dataset and provides summary statistics for:
1. File shapes - consistency check
2. Voxel spacing - uniformity across scanners
3. Data types - storage format
4. Intensity ranges - normalization needs
5. Affine matrices - alignment verification
6. Segmentation labels - missing label analysis
7. Data quality issues - what to fix

Usage:
python3 inspect_dataset.py data/raw/BraTS24/GLI-Challenge-TrainingData
python3 inspect_dataset.py data/raw/BraTS-Africa/95_Glioma

"""

import nibabel as nib
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List
import sys
import warnings
warnings.filterwarnings('ignore')


class DatasetInspector:
    """Comprehensive dataset-wide analysis"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "BraTS"):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        
        # Statistics collectors
        self.stats = {
            'total_cases': 0,
            'shapes': defaultdict(int),
            'spacings': defaultdict(int),
            'dtypes': defaultdict(int),
            'intensities': {mod: [] for mod in ['t1n', 't1c', 't2w', 't2f']},
            'labels_found': defaultdict(int),
            'labels_missing': defaultdict(int),
            'label_combinations': defaultdict(int),
            'missing_files': defaultdict(int),
            'affine_samples': []
        }
    
    def inspect(self):
        """Run complete inspection"""
        print("=" * 80)
        print(f"DATASET INSPECTION: {self.dataset_name}")
        print("=" * 80)
        print(f"Path: {self.dataset_path}\n")
        
        # Find all cases
        cases = self._find_cases()
        self.stats['total_cases'] = len(cases)
        
        print(f"📊 Scanning {len(cases)} cases...")
        
        # Process all cases
        for i, case_dir in enumerate(cases, 1):
            if i % 100 == 0 or i == len(cases):
                print(f"   Progress: {i}/{len(cases)} ({100*i//len(cases)}%)", end='\r')
            
            self._process_case(case_dir, sample_affine=(i <= 3))
        
        print(f"\n✓ Scan complete!\n")
        
        # Print summary
        self._print_summary()
        
        # Save detailed results
        self._save_results()
    
    def _find_cases(self) -> List[Path]:
        """Find all BraTS case directories"""
        cases = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        cases = [d for d in cases if list(d.glob('*-t1c.nii*'))]
        return sorted(cases)
    
    def _process_case(self, case_dir: Path, sample_affine: bool = False):
        """Process one case"""
        # Process modalities
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            files = list(case_dir.glob(f'*-{mod}.nii*'))
            
            if not files:
                self.stats['missing_files'][mod] += 1
                continue
            
            try:
                img = nib.load(str(files[0]))
                data = img.get_fdata()
                
                # Shape
                self.stats['shapes'][str(img.shape)] += 1
                
                # Spacing
                spacing = tuple(round(x, 1) for x in img.header.get_zooms()[:3])
                self.stats['spacings'][str(spacing)] += 1
                
                # Data type
                self.stats['dtypes'][str(img.get_data_dtype())] += 1
                
                # Intensities (brain only)
                brain_mask = data > 0
                if brain_mask.sum() > 0:
                    self.stats['intensities'][mod].append({
                        'min': float(data[brain_mask].min()),
                        'max': float(data[brain_mask].max()),
                        'mean': float(data[brain_mask].mean()),
                        'std': float(data[brain_mask].std())
                    })
                
                # Sample affine
                if sample_affine and mod == 't1c':
                    self.stats['affine_samples'].append({
                        'case': case_dir.name,
                        'affine': img.affine.tolist()
                    })
                    
            except Exception as e:
                pass
        
        # Process segmentation
        seg_files = list(case_dir.glob('*-seg.nii*'))
        
        if not seg_files:
            self.stats['missing_files']['seg'] += 1
            return
        
        try:
            seg = nib.load(str(seg_files[0]))
            seg_data = seg.get_fdata()
            labels = sorted([int(x) for x in np.unique(seg_data)])
            
            # Track label presence
            for label in labels:
                self.stats['labels_found'][label] += 1
            
            # Track missing labels
            if 1 not in labels:
                self.stats['labels_missing']['NCR (label 1)'] += 1
            if 2 not in labels:
                self.stats['labels_missing']['ED (label 2)'] += 1
            if 4 not in labels:
                self.stats['labels_missing']['ET (label 4)'] += 1
            
            # Track label combinations
            label_combo = str(labels)
            self.stats['label_combinations'][label_combo] += 1
            
        except Exception as e:
            pass
    
    def _print_summary(self):
        """Print comprehensive summary"""
        total = self.stats['total_cases']
        
        print("=" * 80)
        print("📊 DATASET SUMMARY")
        print("=" * 80)
        print(f"\n✓ Total cases analyzed: {total}\n")
        
        # 1. SHAPES
        print("1️⃣  FILE SHAPES")
        print("-" * 80)
        for shape, count in sorted(self.stats['shapes'].items(), key=lambda x: -x[1]):
            pct = 100 * count / (total * 4)  # 4 modalities
            status = "✓" if shape == "(182, 218, 182)" else "⚠️"
            print(f"   {status} {shape:25s}: {count:4d} files ({pct:5.1f}%)")
        
        expected_shape = "(182, 218, 182)"
        if expected_shape in self.stats['shapes']:
            if self.stats['shapes'][expected_shape] == total * 4:
                print(f"\n   ✅ All files have correct shape {expected_shape}")
            else:
                print(f"\n   ⚠️  ISSUE: Not all files are {expected_shape}!")
        else:
            print(f"\n   ❌ CRITICAL: No files have expected shape {expected_shape}!")
        
        # 2. VOXEL SPACING
        print(f"\n2️⃣  VOXEL SPACING (mm)")
        print("-" * 80)
        for spacing, count in sorted(self.stats['spacings'].items(), key=lambda x: -x[1])[:5]:
            pct = 100 * count / (total * 4)
            status = "✓" if "(1.0, 1.0, 1.0)" in spacing else "⚠️"
            print(f"   {status} {spacing:25s}: {count:4d} files ({pct:5.1f}%)")
        
        if "(1.0, 1.0, 1.0)" in self.stats['spacings']:
            if self.stats['spacings']["(1.0, 1.0, 1.0)"] == total * 4:
                print(f"\n   ✅ All files have 1mm³ isotropic spacing")
            else:
                print(f"\n   ⚠️  Some files have non-standard spacing")
        
        # 3. DATA TYPES
        print(f"\n3️⃣  DATA TYPES")
        print("-" * 80)
        for dtype, count in sorted(self.stats['dtypes'].items(), key=lambda x: -x[1]):
            pct = 100 * count / (total * 4)
            print(f"   • {dtype:15s}: {count:4d} files ({pct:5.1f}%)")
        
        # 4. INTENSITY RANGES
        print(f"\n4️⃣  INTENSITY RANGES (Brain Voxels Only)")
        print("-" * 80)
        print(f"   {'Modality':<6s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} {'Status'}")
        print("   " + "-" * 70)
        
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            if not self.stats['intensities'][mod]:
                print(f"   {mod.upper():<6s} {'No data':>10s}")
                continue
            
            mins = [x['min'] for x in self.stats['intensities'][mod]]
            maxs = [x['max'] for x in self.stats['intensities'][mod]]
            means = [x['mean'] for x in self.stats['intensities'][mod]]
            stds = [x['std'] for x in self.stats['intensities'][mod]]
            
            min_range = f"{np.min(mins):.0f}-{np.max(mins):.0f}"
            max_range = f"{np.min(maxs):.0f}-{np.max(maxs):.0f}"
            mean_avg = f"{np.mean(means):.0f}"
            std_avg = f"{np.mean(stds):.0f}"
            
            # Check if needs normalization
            if np.max(maxs) > 5000:
                status = "⚠️ High (needs norm)"
            else:
                status = "✓ Normal"
            
            print(f"   {mod.upper():<6s} {min_range:>10s} {max_range:>10s} {mean_avg:>10s} {std_avg:>10s} {status}")
        
        if any(np.max([x['max'] for x in self.stats['intensities'][mod]]) > 5000 
               for mod in ['t1n', 't1c', 't2w', 't2f'] if self.stats['intensities'][mod]):
            print(f"\n   ⚠️  High intensity values detected!")
            print(f"      → Requires Z-score normalization before training")
        else:
            print(f"\n   ✓ Intensity ranges appear reasonable")
        
        # 5. SEGMENTATION LABELS
        print(f"\n5️⃣  SEGMENTATION LABELS")
        print("-" * 80)
        print(f"   Label presence across {total} cases:\n")
        
        for label in sorted(self.stats['labels_found'].keys()):
            count = self.stats['labels_found'][label]
            pct = 100 * count / total
            
            label_names = {
                0: "Background",
                1: "NCR (Necrotic Core)",
                2: "ED (Edema)",
                3: "⚠️  UNEXPECTED LABEL 3",
                4: "ET (Enhancing Tumor)"
            }
            
            name = label_names.get(label, f"Unknown label {label}")
            status = "⚠️" if label == 3 else "✓"
            
            print(f"   {status} Label {label} - {name:30s}: {count:4d} cases ({pct:5.1f}%)")
        
        # Check for label 3
        if 3 in self.stats['labels_found']:
            print(f"\n   ❌ CRITICAL: Label 3 detected in {self.stats['labels_found'][3]} cases!")
            print(f"      → BraTS standard uses labels [0, 1, 2, 4]")
            print(f"      → Label 3 should be remapped to 4 (ET)")
        
        if 4 not in self.stats['labels_found']:
            print(f"\n   ❌ CRITICAL: No cases have label 4 (ET)!")
            print(f"      → All enhancing tumor labeled as 3 instead")
        
        # 6. MISSING LABELS
        print(f"\n6️⃣  MISSING LABELS (Sparse Supervision)")
        print("-" * 80)
        
        if self.stats['labels_missing']:
            for label_name, count in sorted(self.stats['labels_missing'].items(), key=lambda x: -x[1]):
                pct = 100 * count / total
                print(f"   • {label_name:20s}: {count:4d} cases ({pct:5.1f}%) missing this label")
            
            print(f"\n   ℹ️  Missing labels are NORMAL in BraTS:")
            print(f"      - Not all tumors have necrosis (NCR)")
            print(f"      - Not all tumors have enhancement (ET)")
            print(f"      - Edema (ED) is almost always present")
        else:
            print(f"   ✓ All expected labels found in all cases")
        
        # 7. LABEL COMBINATIONS
        print(f"\n7️⃣  COMMON LABEL COMBINATIONS (Top 10)")
        print("-" * 80)
        
        for combo, count in sorted(self.stats['label_combinations'].items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / total
            status = "⚠️" if '3' in combo else "✓"
            print(f"   {status} {combo:35s}: {count:4d} cases ({pct:5.1f}%)")
        
        # 8. MISSING FILES
        print(f"\n8️⃣  MISSING FILES")
        print("-" * 80)
        
        if any(self.stats['missing_files'].values()):
            for file_type, count in sorted(self.stats['missing_files'].items()):
                if count > 0:
                    pct = 100 * count / total
                    print(f"   ⚠️  {file_type.upper():4s}: {count:4d} cases ({pct:5.1f}%) missing")
        else:
            print(f"   ✓ No missing files detected")
        
        # 9. DATA QUALITY ISSUES
        print(f"\n9️⃣  DATA QUALITY ISSUES")
        print("-" * 80)
        
        issues = []
        
        # Check shape
        if "(240, 240, 155)" in self.stats['shapes']:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Wrong shape detected',
                'count': self.stats['shapes']["(240, 240, 155)"] // 4,
                'fix': 'Resample to (182, 218, 182)'
            })
        
        # Check label 3
        if 3 in self.stats['labels_found']:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Label 3 exists (should be 4)',
                'count': self.stats['labels_found'][3],
                'fix': 'Remap label 3 → 4'
            })
        
        # Check intensities
        high_intensity_mods = []
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            if self.stats['intensities'][mod]:
                if np.max([x['max'] for x in self.stats['intensities'][mod]]) > 5000:
                    high_intensity_mods.append(mod.upper())
        
        if high_intensity_mods:
            issues.append({
                'severity': 'MEDIUM',
                'issue': f'High intensity values in {", ".join(high_intensity_mods)}',
                'count': total,
                'fix': 'Apply Z-score normalization'
            })
        
        # Print issues
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"\n   {i}. [{issue['severity']}] {issue['issue']}")
                print(f"      Affected: {issue['count']} cases")
                print(f"      Fix: {issue['fix']}")
        else:
            print(f"\n   ✅ No critical issues detected!")
        
        # 10. RECOMMENDATIONS
        print(f"\n🔟 RECOMMENDATIONS")
        print("-" * 80)
        
        if "(240, 240, 155)" in self.stats['shapes']:
            print(f"\n   1. FIX SHAPES:")
            print(f"      python fix_africa_dataset.py")
            print(f"      → Resamples (240, 240, 155) → (182, 218, 182)")
        
        if 3 in self.stats['labels_found']:
            print(f"\n   2. FIX LABELS:")
            print(f"      python fix_africa_dataset.py")
            print(f"      → Remaps label 3 → 4")
        
        if high_intensity_mods:
            print(f"\n   3. NORMALIZE INTENSITIES:")
            print(f"      Use Z-score normalization during preprocessing")
            print(f"      → Already implemented in brats_preprocessing.py")
        
        print(f"\n   4. HANDLE SPARSE LABELS:")
        print(f"      Use sparse-aware loss function during training")
        print(f"      → Only compute loss for present labels")
        
        print("\n" + "=" * 80)
        print("✓ Inspection complete!")
        print("=" * 80)
    
    def _save_results(self):
        """Save detailed results to JSON"""
        output_file = f"dataset_inspection_{self.dataset_name}.json"
        
        # Convert to JSON-serializable format
        results = {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path),
            'total_cases': self.stats['total_cases'],
            'shapes': dict(self.stats['shapes']),
            'spacings': dict(self.stats['spacings']),
            'dtypes': dict(self.stats['dtypes']),
            'labels_found': dict(self.stats['labels_found']),
            'labels_missing': dict(self.stats['labels_missing']),
            'label_combinations': dict(self.stats['label_combinations']),
            'missing_files': dict(self.stats['missing_files']),
            'affine_samples': self.stats['affine_samples'][:3],  # Save first 3
            'intensity_summary': {}
        }
        
        # Add intensity summary
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            if self.stats['intensities'][mod]:
                results['intensity_summary'][mod] = {
                    'min_across_cases': float(np.min([x['min'] for x in self.stats['intensities'][mod]])),
                    'max_across_cases': float(np.max([x['max'] for x in self.stats['intensities'][mod]])),
                    'mean_across_cases': float(np.mean([x['mean'] for x in self.stats['intensities'][mod]])),
                    'std_across_cases': float(np.mean([x['std'] for x in self.stats['intensities'][mod]]))
                }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {output_file}")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python inspect_dataset.py <path_to_dataset> [dataset_name]")
        print("\nExamples:")
        print("  python inspect_dataset.py data/raw/BraTS-Africa/95_Glioma Africa")
        print("  python inspect_dataset.py data/raw/BraTS2023-GLI BraTS2023")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else Path(dataset_path).name
    
    inspector = DatasetInspector(dataset_path, dataset_name)
    inspector.inspect()


if __name__ == '__main__':
    main()