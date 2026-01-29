"""Runtime benchmarks for complex workflow examples using pytest-benchmark."""
from .timing import workload_mpra_example


class TestMPRA:
    # Test complex MPRA example
    if True:
        def test_mpra_num_seqs_10(self, benchmark): benchmark(workload_mpra_example, num_seqs=10) # 21 ms
        def test_mpra_num_seqs_30(self, benchmark): benchmark(workload_mpra_example, num_seqs=30) # 45 ms
        def test_mpra_num_seqs_100(self, benchmark): benchmark(workload_mpra_example, num_seqs=100) # 132 ms
        def test_mpra_num_seqs_300(self, benchmark): benchmark(workload_mpra_example, num_seqs=300) # 356 ms
        def test_mpra_num_seqs_1000(self, benchmark): benchmark(workload_mpra_example, num_seqs=1000) # 1169 ms
        #def test_mpra_num_seqs_3000(self, benchmark): benchmark(workload_mpra_example, num_seqs=3000) # 3556 ms
