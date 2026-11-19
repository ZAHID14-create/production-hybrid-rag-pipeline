import sys
import re
from pathlib import Path

# Fix Windows Application Control policy blocking native _regex.pyd
sys.modules['regex'] = re

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

def main():
    print("=" * 60)
    print("RUNNING PHASE 6 EVALUATION TEST SUITE")
    print("=" * 60)
    
    import tests.test_phase6_evaluation as t6
    t6.run_all_tests()
    print("\nPhase 6 Evaluation tests completed successfully!")

if __name__ == "__main__":
    main()
