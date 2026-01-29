import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

INGESTION_SCRIPT = PROJECT_ROOT / "01_Multi_Source_Document_Ingestion" / "ingestion_engine.py"
REQ_UNDERSTANDING_SCRIPT = PROJECT_ROOT / "02_Requirement_Understanding" / "requirement_understanding_engine.py"

CRU_BASE = PROJECT_ROOT / "03_CRU_Normalization"

MODULE_1 = CRU_BASE / "module_1_signals" / "extract_canonical_signals.py"
MODULE_2 = CRU_BASE / "module_2_assembly" / "assemble_candidate_requirements.py"
MODULE_3 = CRU_BASE / "module_3_finalization" / "finalize_crus.py"


def run(cmd, cwd=None):
    print(f"\n▶ Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    # ---- Step 1: Ingestion (interactive input allowed) ----
    run([sys.executable, str(INGESTION_SCRIPT)])

    # ---- Step 2: Requirement Understanding ----
    run(
        [sys.executable, str(REQ_UNDERSTANDING_SCRIPT)],
        cwd=PROJECT_ROOT / "02_Requirement_Understanding"
    )

    # ---- Step 3: CRU Normalization ----
    run([
        sys.executable, str(MODULE_1),
        "--input", "02_Requirement_Understanding/output/requirements_extracted_grouped.json",
        "--output", "03_CRU_Normalization/module_1_signals/output/canonical_signals.json"
    ])

    run([
        sys.executable, str(MODULE_2),
        "--input", "03_CRU_Normalization/module_1_signals/output/canonical_signals.json",
        "--output", "03_CRU_Normalization/module_2_assembly/output/candidate_requirement_assemblies.json"
    ])

    run([
        sys.executable, str(MODULE_3),
        "--input", "03_CRU_Normalization/module_2_assembly/output/candidate_requirement_assemblies.json",
        "--output", "03_CRU_Normalization/module_3_finalization/output/cru_units.json"
    ])

    print("\n✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()
