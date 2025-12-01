"""
Run all sanity checks and tests.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Run all test modules."""
    print("=" * 60)
    print("Running all sanity checks and tests")
    print("=" * 60)
    print()
    
    # Import and run EH36M tests
    print("\n[1/3] EH36M Tests")
    print("-" * 60)
    try:
        from tests.test_eh36m import (
            test_model_forward_pass,
            test_model_loss_computation,
            test_dataset_sample_shapes,
            test_training_stability
        )
        test_model_forward_pass()
        test_model_loss_computation()
        test_dataset_sample_shapes()
        test_training_stability()
    except Exception as e:
        print(f"✗ EH36M tests failed: {e}")
    
    # Import and run MotionBERT tests
    print("\n[2/3] MotionBERT Tests")
    print("-" * 60)
    try:
        from tests.test_motionbert import (
            test_motionbert_forward_pass,
            test_motionbert_input_shapes
        )
        test_motionbert_forward_pass()
        test_motionbert_input_shapes()
    except Exception as e:
        print(f"✗ MotionBERT tests failed: {e}")
    
    # Import and run end-to-end tests
    print("\n[3/3] End-to-End Tests")
    print("-" * 60)
    try:
        from tests.test_end_to_end import (
            test_cache_directory_exists,
            test_build_events_tensor,
            test_pose_sequence_shapes,
            test_motionbert_input_preparation
        )
        test_cache_directory_exists()
        test_build_events_tensor()
        test_pose_sequence_shapes()
        test_motionbert_input_preparation()
    except Exception as e:
        print(f"✗ End-to-end tests failed: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

