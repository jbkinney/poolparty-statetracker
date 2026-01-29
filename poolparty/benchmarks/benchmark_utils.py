"""Utilities for generating benchmark tests dynamically."""


def generate_benchmark_tests(specs: dict) -> dict:
    """
    Generate pytest benchmark test classes from a specification dict.
    
    Args:
        specs: Dict mapping class names to lists of benchmark specs.
               Each spec is (workload_fn, param_name, values, constants_dict, enabled).
    
    Returns:
        Dict of class names to class objects to inject into globals().
    """
    classes = {}
    for class_name, benchmarks in specs.items():
        methods = {}
        for workload, param, values, constants, enabled in benchmarks:
            if not enabled:
                continue
            
            for val in values:
                # Create test name including constants and variable param
                val_str = str(val).replace(".", "_")
                const_parts = [f"{k}_{str(v).replace('.', '_')}" for k, v in constants.items()]
                param_part = f"{param}_{val_str}"
                name_parts = [workload.__name__] + const_parts + [param_part]
                test_name = "test_" + "_".join(name_parts)
                
                # Variable param overrides constants
                all_kwargs = {**constants, param: val}
                
                def make_test(w, kwargs):
                    def test(self, benchmark):
                        benchmark(w, **kwargs)
                    return test
                
                methods[test_name] = make_test(workload, all_kwargs)
        
        classes[f"{class_name}"] = type(f"{class_name}", (), methods)
    return classes
