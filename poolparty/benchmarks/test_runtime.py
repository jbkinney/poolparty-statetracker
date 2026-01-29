"""Runtime benchmarks using pytest-benchmark."""
import pytest
from .workloads import (
    workload_mutagenize_num_mut,
    workload_mutagenize_mut_rate,
    workload_shuffle_seq,
    workload_deletion_scan,
    workload_insertion_scan,
    workload_get_kmers,
    workload_from_iupac,
)

class TestMutagenize:

    # Test num mutations: (seq_len = 100, num_seqs=100, mode='random'). Conclusion: roughly constant until ~100
    if False:
        def test_mutagenize_num_mut_1(self, benchmark): benchmark(workload_mutagenize_num_mut, num_mut=1) # 8 ms
        def test_mutagenize_num_mut_3(self, benchmark): benchmark(workload_mutagenize_num_mut, num_mut=3) # 8 ms
        def test_mutagenize_num_mut_10(self, benchmark): benchmark(workload_mutagenize_num_mut, num_mut=10) # 9 ms
        def test_mutagenize_num_mut_30(self, benchmark): benchmark(workload_mutagenize_num_mut, num_mut=30) # 11 ms
        def test_mutagenize_num_mut_100(self, benchmark): benchmark(workload_mutagenize_num_mut, num_mut=100) # 19 ms
            
    # Test sequence length (mut_rate=0.10, num_seqs=100). Conclusion: roughly linear above 1_000 nt
    if False:
        def test_mutagenize_seq_len_10(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=10) # 7 ms
        def test_mutagenize_seq_len_30(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=30) # 7 ms
        def test_mutagenize_seq_len_100(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=100) # 10 ms
        def test_mutagenize_seq_len_300(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=300) # 16 ms
        def test_mutagenize_seq_len_1000(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=1_000) # 34 ms 
        def test_mutagenize_seq_len_3000(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=3_000) # 100 ms
        def test_mutagenize_seq_len_10000(self, benchmark): benchmark(workload_mutagenize_mut_rate, seq_len=10_000) # 324 ms

    # Test mutation rate (seq_length=100, num_seqs=100). Conclusion: roughly constant until ~100% 
    if True:
        def test_mutagenize_mut_rate_1(self, benchmark): benchmark(workload_mutagenize_mut_rate, mut_rate=0.01) # 8 ms
        def test_mutagenize_mut_rate_3(self, benchmark): benchmark(workload_mutagenize_mut_rate, mut_rate=0.03) # 8 ms
        def test_mutagenize_mut_rate_10(self, benchmark): benchmark(workload_mutagenize_mut_rate, mut_rate=0.10) # 9 ms
        def test_mutagenize_mut_rate_30(self, benchmark): benchmark(workload_mutagenize_mut_rate, mut_rate=0.30) # 12 ms
        def test_mutagenize_mut_rate_100(self, benchmark): benchmark(workload_mutagenize_mut_rate, mut_rate=1.00) # 19 ms
      
        
class TestShuffle:
    # Test sequence length (num_seqs=100)
    if True:
        def test_shuffle_seq_len_10(self, benchmark): benchmark(workload_shuffle_seq, seq_len=10) # 6 ms
        def test_shuffle_seq_len_30(self, benchmark): benchmark(workload_shuffle_seq, seq_len=30) # 6 ms
        def test_shuffle_seq_len_100(self, benchmark): benchmark(workload_shuffle_seq, seq_len=100) # 9 ms
        def test_shuffle_seq_len_300(self, benchmark): benchmark(workload_shuffle_seq, seq_len=300) # 17 ms
        def test_shuffle_seq_len_1000(self, benchmark): benchmark(workload_shuffle_seq, seq_len=1000) # 50 ms
        
        
class TestDeletionScan:
    # Test sequence length (num_seqs=100): roughly constant until ~300 nt
    if True:
        def test_deletion_scan_10(self, benchmark): benchmark(workload_deletion_scan, seq_len=10) # 11 ms
        def test_deletion_scan_30(self, benchmark): benchmark(workload_deletion_scan, seq_len=30) # 11 ms
        def test_deletion_scan_100(self, benchmark): benchmark(workload_deletion_scan, seq_len=100) # 14 ms
        def test_deletion_scan_300(self, benchmark): benchmark(workload_deletion_scan, seq_len=300) # 20 ms
        def test_deletion_scan_1000(self, benchmark): benchmark(workload_deletion_scan, seq_len=1000) # 45 ms
        
        
class TestInsertionScan:
    # Test sequence length (num_seqs=100, ins_len=5): roughly constant until ~300 nt
    if False:
        def test_insertion_scan_seq_len_10(self, benchmark): benchmark(workload_insertion_scan, seq_len=10) # 9 ms
        def test_insertion_scan_seq_len_30(self, benchmark): benchmark(workload_insertion_scan, seq_len=30) # 10 ms
        def test_insertion_scan_seq_len_100(self, benchmark): benchmark(workload_insertion_scan, seq_len=100) # 12 ms
        def test_insertion_scan_seq_len_300(self, benchmark): benchmark(workload_insertion_scan, seq_len=300) # 19 ms
        def test_insertion_scan_seq_len_1000(self, benchmark): benchmark(workload_insertion_scan, seq_len=1000) # 44 ms

    # Test sequence length (num_seqs=100, seq_len=100): roughtly constant. 
    if True:
        def test_insertion_scan_ins_len_1(self, benchmark): benchmark(workload_insertion_scan, ins_len=1) # 13 ms
        def test_insertion_scan_ins_len_3(self, benchmark): benchmark(workload_insertion_scan, ins_len=3) # 13 ms
        def test_insertion_scan_ins_len_10(self, benchmark): benchmark(workload_insertion_scan, ins_len=10) # 13 ms
        def test_insertion_scan_ins_len_30(self, benchmark): benchmark(workload_insertion_scan, ins_len=30) # 13 ms
        def test_insertion_scan_ins_len_100(self, benchmark): benchmark(workload_insertion_scan, ins_len=100)  # 14 ms
        
        
class TestGetKmers:
    
    # Test kmer length (num_seqs=100): roughly constant and very fast. 
    if True:
        def test_get_kmers_kmer_len_1(self, benchmark): benchmark(workload_get_kmers, kmer_len=1) # 2.5 ms
        def test_get_kmers_kmer_len_3(self, benchmark): benchmark(workload_get_kmers, kmer_len=3) # 2.5 ms
        def test_get_kmers_kmer_len_10(self, benchmark): benchmark(workload_get_kmers, kmer_len=10) # 2.6 ms
        def test_get_kmers_kmer_len_30(self, benchmark): benchmark(workload_get_kmers, kmer_len=30) # 2.8 ms
        def test_get_kmers_kmer_len_100(self, benchmark): benchmark(workload_get_kmers, kmer_len=100) # 3.4 ms
        

class TestFromIupac:
    # Test iupac sequence length (num_seqs=100): roughly constant and very fast. 
    if True:
        def test_from_iupac_seq_len_1(self, benchmark): benchmark(workload_from_iupac, seq_len=1) # 2.1 ms
        def test_from_iupac_seq_len_3(self, benchmark): benchmark(workload_from_iupac, seq_len=3) # 2.1 ms
        def test_from_iupac_seq_len_10(self, benchmark): benchmark(workload_from_iupac, seq_len=10) # 2.2 ms
        def test_from_iupac_seq_len_30(self, benchmark): benchmark(workload_from_iupac, seq_len=30) # 2.5 ms
        def test_from_iupac_seq_len_100(self, benchmark): benchmark(workload_from_iupac, seq_len=100) # 2.7 ms