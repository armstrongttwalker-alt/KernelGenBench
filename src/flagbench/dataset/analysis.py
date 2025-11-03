from kernel_list import op_name_list, IMPL_INFO, PYTORCH_OPERATORS

def analyze_operators():
    impl_info_keys = set(IMPL_INFO.keys())
    pytorch_operators_keys = set(key.split('.')[-1] for key in PYTORCH_OPERATORS.keys())

    matched_keys = impl_info_keys.intersection(pytorch_operators_keys)
    unmatched_impl_info_keys = impl_info_keys - pytorch_operators_keys
    unmatched_pytorch_operators_keys = pytorch_operators_keys - impl_info_keys

    print(f"\n--- Operator Analysis ---")
    print(f"IMPL_INFO keys count: {len(impl_info_keys)}")
    print(f"PYTORCH_OPERATORS keys count (after stripping 'torch.'): {len(pytorch_operators_keys)}")

    print(f"\nMatched keys ({len(matched_keys)}):")
    for key in sorted(list(matched_keys)):
        print(f"  - {key}")

    print(f"\nUnmatched IMPL_INFO keys ({len(unmatched_impl_info_keys)}):")
    for key in sorted(list(unmatched_impl_info_keys)):
        print(f"  - {key}")

    print(f"\nUnmatched PYTORCH_OPERATORS keys ({len(unmatched_pytorch_operators_keys)}):")
    for key in sorted(list(unmatched_pytorch_operators_keys)):
        print(f"  - {key}")

    print(f"\n--- Analysis Complete ---")

if __name__ == '__main__':
    analyze_operators()