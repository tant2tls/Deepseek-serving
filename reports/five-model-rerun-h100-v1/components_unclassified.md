# Largest kernels left in `other_elementwise` (µs summed over the averaged steps, all ranks)

## v4-0731 decode128k_B1

- `Kernel`: 130321.143
- `memcpy32_post`: 30456.592
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 21902.872999999996
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9923.425999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3077.389
- `hc_head_fuse_tilelang_kernel`: 2764.741
- `_gumbel_sample_kernel`: 2679.027
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2636.741
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2632.872
- `_compute_swa_indices_and_lens_kernel`: 1852.4800000000002
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1832.922
- `_post_update_kernel`: 1452.732

## v4-0731 decode128k_B8

- `Kernel`: 138272.55100000004
- `memcpy32_post`: 29054.441
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27968.74
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9553.364
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 3674.909
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3304.2490000000003
- `_gumbel_sample_kernel`: 3055.0330000000004
- `hc_head_fuse_tilelang_kernel`: 2790.338
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1943.981
- `_compute_swa_indices_and_lens_kernel`: 1859.3609999999999
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1732.965
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1685.1200000000001

## v4-0731 decode16k_B1

- `Kernel`: 131622.63799999995
- `memcpy32_post`: 31130.212
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 22201.452999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 10123.546
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3120.0870000000004
- `_gumbel_sample_kernel`: 2835.905
- `hc_head_fuse_tilelang_kernel`: 2816.6800000000003
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2718.553
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2693.7500000000005
- `_compute_swa_indices_and_lens_kernel`: 1947.855
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1854.736
- `_post_update_kernel`: 1471.802

## v4-0731 decode16k_B8

- `Kernel`: 137185.583
- `memcpy32_post`: 30156.072999999997
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 29015.54499999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 10321.856999999996
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 3761.0870000000004
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3516.372999999999
- `_gumbel_sample_kernel`: 3366.918
- `hc_head_fuse_tilelang_kernel`: 2903.303
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2025.3339999999998
- `_compute_swa_indices_and_lens_kernel`: 1933.1099999999997
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1795.6039999999998
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1750.471

## v4-0731 decode1k_B1

- `Kernel`: 130283.45999999999
- `memcpy32_post`: 30497.108
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 21835.101
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9922.922000000004
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3093.809
- `_gumbel_sample_kernel`: 2836.9879999999994
- `hc_head_fuse_tilelang_kernel`: 2792.0110000000004
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2690.712
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2661.5339999999997
- `_compute_swa_indices_and_lens_kernel`: 1938.5999999999997
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1846.1879999999999
- `_post_update_kernel`: 1459.103

## v4-0731 decode1k_B8

- `Kernel`: 129196.48
- `memcpy32_post`: 28318.798
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27299.871999999996
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9360.7
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 3569.729
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3337.206
- `_gumbel_sample_kernel`: 2863.396
- `hc_head_fuse_tilelang_kernel`: 2735.7350000000006
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1901.3889999999997
- `_compute_swa_indices_and_lens_kernel`: 1807.249
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1696.65
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1649.877

## v4-0731 decode64k_B1

- `Kernel`: 129896.62899999997
- `memcpy32_post`: 30463.918999999987
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 21748.439
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9903.836999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3057.7209999999995
- `hc_head_fuse_tilelang_kernel`: 2760.441
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2648.819
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2618.7309999999998
- `_gumbel_sample_kernel`: 2616.392
- `_compute_swa_indices_and_lens_kernel`: 1832.6930000000002
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1821.5939999999996
- `_post_update_kernel`: 1452.3609999999999

## v4-0731 decode64k_B8

- `Kernel`: 134185.83800000002
- `memcpy32_post`: 29080.950000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27876.717000000008
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9541.128000000004
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 3645.827
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3391.075
- `_gumbel_sample_kernel`: 3037.8269999999998
- `hc_head_fuse_tilelang_kernel`: 2785.8469999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1946.622
- `_compute_swa_indices_and_lens_kernel`: 1837.94
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1735.1730000000002
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1683.5470000000003

## v4-0731 prefill128k

- `Kernel`: 120265.937
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 46433.36
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 6615.851000000001
- `hc_head_fuse_tilelang_kernel`: 3911.5179999999996
- `_apply_write_kernel`: 1726.2289999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1183.659
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 456.049
- `_compute_swa_indices_and_lens_kernel`: 423.99
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 346.446
- `_prepare_prefill_inputs_kernel`: 238.25
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 116.133
- `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::arra`: 50.05199999999999

## v4-0731 prefill16k

- `Kernel`: 116705.27000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 46408.544
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4949.662
- `hc_head_fuse_tilelang_kernel`: 3842.838
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1190.848
- `_apply_write_kernel`: 636.85
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 445.244
- `_compute_swa_indices_and_lens_kernel`: 411.582
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 342.236
- `_prepare_prefill_inputs_kernel`: 216.70199999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 114.59000000000002
- `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::arra`: 49.298

## v4-0731 prefill1k

- `Kernel`: 17950.760000000002
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 11253.135999999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 4980.078
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1183.9280000000003
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 1011.544
- `hc_head_fuse_tilelang_kernel`: 577.0160000000001
- `_apply_write_kernel`: 288.386
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 142.75199999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 115.22999999999999
- `_compute_swa_indices_and_lens_kernel`: 96.65
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 78.224
- `_prepare_prefill_inputs_kernel`: 64.656

## v4-0731 prefill64k

- `Kernel`: 117817.157
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 46416.724
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5691.535000000001
- `hc_head_fuse_tilelang_kernel`: 3869.887
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1195.0180000000003
- `_apply_write_kernel`: 963.1750000000001
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 451.156
- `_compute_swa_indices_and_lens_kernel`: 420.754
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 345.2939999999999
- `_prepare_prefill_inputs_kernel`: 239.97499999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 116.554
- `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::arra`: 49.604

## v41 decode128k_B1

- `Kernel`: 47488.532
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20292.604
- `memcpy32_post`: 17758.270999999997
- `_candidate_flags_kernel`: 7525.803000000001
- `_compute_swa_indices_and_lens_kernel`: 5628.778
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4011.576
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3438.6320000000005
- `_gumbel_sample_kernel`: 3227.268
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3181.129
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 2663.592
- `_hash_ids_kernel`: 1822.187
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1796.675

## v41 decode128k_B8

- `Kernel`: 52366.452000000005
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26475.422
- `memcpy32_post`: 16247.884
- `_candidate_flags_kernel`: 8328.886999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 6828.574000000001
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 6787.912000000001
- `_compute_swa_indices_and_lens_kernel`: 6018.7970000000005
- `_gumbel_sample_kernel`: 3908.437
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3572.7830000000004
- `_hash_ids_kernel`: 2031.009
- `_combine_sampled_and_draft_tokens_kernel`: 2015.6209999999999
- `_get_num_sampled_and_rejected_kernel`: 1992.411

## v41 decode16k_B1

- `Kernel`: 47535.591
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20323.283000000003
- `memcpy32_post`: 17763.700999999997
- `_candidate_flags_kernel`: 6377.721
- `_compute_swa_indices_and_lens_kernel`: 5621.558999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4019.989
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3445.1169999999997
- `_gumbel_sample_kernel`: 3183.7490000000003
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3181.7729999999997
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 2668.0550000000003
- `_hash_ids_kernel`: 1823.2939999999999
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1800.3419999999999

## v41 decode16k_B8

- `Kernel`: 53648.206
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27516.588000000003
- `memcpy32_post`: 16908.439
- `_candidate_flags_kernel`: 7114.53
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 7111.43
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 7096.601000000001
- `_compute_swa_indices_and_lens_kernel`: 6291.089
- `_gumbel_sample_kernel`: 4059.7450000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3715.935
- `_hash_ids_kernel`: 2125.251
- `_combine_sampled_and_draft_tokens_kernel`: 2028.3760000000002
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2012.3229999999999

## v41 decode1k_B1

- `Kernel`: 47579.03199999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20311.232999999997
- `memcpy32_post`: 17719.530000000006
- `_compute_swa_indices_and_lens_kernel`: 6007.905000000001
- `_candidate_flags_kernel`: 5027.404
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4026.6770000000006
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3481.264000000001
- `_gumbel_sample_kernel`: 3460.682
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3194.003
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 2697.4610000000002
- `_hash_ids_kernel`: 1853.7459999999996
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1807.8450000000003

## v41 decode1k_B8

- `Kernel`: 51177.242000000006
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26475.316000000003
- `memcpy32_post`: 16261.135999999995
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 6829.087000000001
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 6789.537
- `_compute_swa_indices_and_lens_kernel`: 6010.034000000001
- `_candidate_flags_kernel`: 5287.208999999999
- `_gumbel_sample_kernel`: 3976.5260000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3574.554000000001
- `_hash_ids_kernel`: 2038.0520000000001
- `_combine_sampled_and_draft_tokens_kernel`: 2028.1380000000001
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1938.6570000000002

## v41 decode64k_B1

- `Kernel`: 47508.030999999995
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20304.561999999998
- `memcpy32_post`: 17750.590999999993
- `_candidate_flags_kernel`: 7291.739
- `_compute_swa_indices_and_lens_kernel`: 5800.3859999999995
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4004.7600000000007
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3406.202
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3189.4939999999997
- `_gumbel_sample_kernel`: 3184.8970000000004
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 2637.248
- `_hash_ids_kernel`: 1813.6950000000002
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1795.796

## v41 decode64k_B8

- `Kernel`: 52405.877000000015
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26480.505999999998
- `memcpy32_post`: 16253.764999999996
- `_candidate_flags_kernel`: 8351.408000000003
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 6839.531999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 6822.483
- `_compute_swa_indices_and_lens_kernel`: 6011.902999999999
- `_gumbel_sample_kernel`: 3937.3769999999995
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 3570.518
- `_hash_ids_kernel`: 2039.5189999999998
- `_combine_sampled_and_draft_tokens_kernel`: 1947.2949999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1939.35

## v41 prefill128k

- `Kernel`: 46527.261000000006
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 31021.22
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27463.444000000003
- `_hc_head_reduce_store_kernel`: 2267.51
- `memcpy128`: 2108.012
- `_apply_write_kernel`: 1786.3249999999998
- `_compute_swa_indices_and_lens_kernel`: 1691.484
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1671.358
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1130.631
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 519.9549999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 490.12900000000013
- `memcpy32_post`: 451.3600000000001

## v41 prefill16k

- `Kernel`: 45229.614
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 32278.958
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27465.972000000005
- `_hc_head_reduce_store_kernel`: 2261.772
- `memcpy128`: 2108.092
- `_compute_swa_indices_and_lens_kernel`: 1681.274
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1667.7200000000003
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1119.498
- `_apply_write_kernel`: 667.152
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 526.786
- `memcpy32_post`: 429.95
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 410.612

## v41 prefill1k

- `Kernel`: 8296.620000000003
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 3479.528
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 3361.206000000001
- `memcpy128`: 2086.0060000000008
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1106.1780000000003
- `_compute_swa_indices_and_lens_kernel`: 569.2139999999999
- `memcpy32_post`: 433.2399999999999
- `_apply_write_kernel`: 312.20799999999997
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 310.28
- `_hc_head_reduce_store_kernel`: 248.174
- `void at::native::index_elementwise_kernel<128, 4, at::native::index_copy_kernel_impl<at::n`: 232.106
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 212.25799999999998

## v41 prefill64k

- `Kernel`: 47288.526
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 30071.805
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27454.481000000003
- `_hc_head_reduce_store_kernel`: 2266.9139999999998
- `memcpy128`: 2106.6110000000003
- `_compute_swa_indices_and_lens_kernel`: 1695.814
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1674.6
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1122.227
- `_apply_write_kernel`: 1000.8589999999999
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 517.398
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 442.6400000000001
- `memcpy32_post`: 439.4170000000001

## mimo-v26 decode128k_B1

- `triton_poi_fused_0`: 29954.599999999995
- `memcpy32_post`: 19814.220999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3301.0189999999993
- `_gumbel_sample_kernel`: 2636.9819999999995
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1853.562
- `_post_update_kernel`: 1579.761
- `triton_poi_fused_1`: 1495.8959999999997
- `_combine_sampled_and_draft_tokens_kernel`: 1198.5320000000002
- `_get_num_sampled_and_rejected_kernel`: 1152.341
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 850.8000000000001
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 787.912
- `_prepare_pos_seq_lens_kernel`: 595.2620000000001

## mimo-v26 decode128k_B8

- `triton_poi_fused_0`: 31955.9
- `memcpy32_post`: 18714.652
- `_gumbel_sample_kernel`: 2919.0470000000005
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2007.3119999999997
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1886.297
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1725.798
- `triton_poi_fused_1`: 1524.698
- `_post_update_kernel`: 1427.661
- `_combine_sampled_and_draft_tokens_kernel`: 1284.6079999999997
- `_get_num_sampled_and_rejected_kernel`: 1211.8029999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 805.8679999999999
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 720.6270000000002

## mimo-v26 decode16k_B1

- `triton_poi_fused_0`: 32025.915000000005
- `memcpy32_post`: 21021.796999999995
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3492.622
- `_gumbel_sample_kernel`: 2790.41
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1965.5539999999999
- `_post_update_kernel`: 1686.2710000000002
- `triton_poi_fused_1`: 1631.207
- `_combine_sampled_and_draft_tokens_kernel`: 1255.8159999999998
- `_get_num_sampled_and_rejected_kernel`: 1214.6159999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 901.0469999999997
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 835.267
- `_prepare_pos_seq_lens_kernel`: 630.442

## mimo-v26 decode16k_B8

- `triton_poi_fused_0`: 31893.497
- `memcpy32_post`: 19135.930999999997
- `_gumbel_sample_kernel`: 2875.913
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2049.254
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1917.112
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1722.1580000000004
- `triton_poi_fused_1`: 1542.6370000000002
- `_post_update_kernel`: 1459.7410000000002
- `_combine_sampled_and_draft_tokens_kernel`: 1322.7259999999999
- `_get_num_sampled_and_rejected_kernel`: 1230.2229999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 820.645
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 733.1789999999999

## mimo-v26 decode1k_B1

- `triton_poi_fused_0`: 32603.449999999993
- `memcpy32_post`: 21564.167999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3582.5239999999994
- `_gumbel_sample_kernel`: 2973.461
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2009.9030000000002
- `_post_update_kernel`: 1695.1740000000002
- `triton_poi_fused_1`: 1616.1830000000002
- `_combine_sampled_and_draft_tokens_kernel`: 1290.3159999999998
- `_get_num_sampled_and_rejected_kernel`: 1238.6119999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 924.2299999999996
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 857.1379999999998
- `_prepare_pos_seq_lens_kernel`: 644.7750000000001

## mimo-v26 decode1k_B8

- `triton_poi_fused_0`: 33786.986000000004
- `memcpy32_post`: 20285.190999999995
- `_gumbel_sample_kernel`: 3164.9439999999995
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2168.6859999999997
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 2027.1439999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1823.576
- `triton_poi_fused_1`: 1657.719
- `_post_update_kernel`: 1565.2640000000001
- `_combine_sampled_and_draft_tokens_kernel`: 1396.1329999999998
- `_get_num_sampled_and_rejected_kernel`: 1302.506
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 870.117
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 775.9300000000003

## mimo-v26 decode64k_B1

- `triton_poi_fused_0`: 32797.464
- `memcpy32_post`: 21767.987999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3621.9469999999997
- `_gumbel_sample_kernel`: 2870.218
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2028.018
- `_post_update_kernel`: 1747.3529999999998
- `triton_poi_fused_1`: 1624.2479999999998
- `_combine_sampled_and_draft_tokens_kernel`: 1310.3210000000001
- `_get_num_sampled_and_rejected_kernel`: 1256.443
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 932.3479999999997
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 864.421
- `_prepare_pos_seq_lens_kernel`: 650.775

## mimo-v26 decode64k_B8

- `triton_poi_fused_0`: 32676.391000000007
- `memcpy32_post`: 19475.572
- `_gumbel_sample_kernel`: 2897.6430000000005
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2082.74
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1956.499
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1754.43
- `triton_poi_fused_1`: 1585.9410000000003
- `_post_update_kernel`: 1487.864
- `_combine_sampled_and_draft_tokens_kernel`: 1352.131
- `_get_num_sampled_and_rejected_kernel`: 1252.56
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 837.5280000000001
- `_apply_write_kernel`: 785.849

## mimo-v26 prefill128k

- `triton_poi_fused_2`: 16658.027
- `_apply_write_kernel`: 1616.5939999999998
- `triton_poi_fused_mul_silu_slice_1`: 504.442
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 400.216
- `triton_poi_fused_3`: 362.836
- `triton_poi_fused_1`: 358.4189999999999
- `_prepare_prefill_inputs_kernel`: 233.707
- `_combine_sampled_and_draft_tokens_kernel`: 39.748000000000005
- `_prepare_pos_seq_lens_kernel`: 38.115
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.64
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 16.258
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.904

## mimo-v26 prefill16k

- `triton_poi_fused_2`: 16563.420000000002
- `_apply_write_kernel`: 619.4179999999999
- `triton_poi_fused_mul_silu_slice_1`: 506.37
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 407.586
- `triton_poi_fused_1`: 357.33600000000007
- `triton_poi_fused_3`: 357.336
- `_prepare_prefill_inputs_kernel`: 216.75400000000002
- `_combine_sampled_and_draft_tokens_kernel`: 39.498
- `_prepare_pos_seq_lens_kernel`: 36.172
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 15.94
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 15.748000000000001
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.552

## mimo-v26 prefill1k

- `triton_poi_fused_2`: 2234.9199999999996
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1627.9560000000001
- `_apply_write_kernel`: 280.35200000000003
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 66.96
- `_prepare_prefill_inputs_kernel`: 64.78800000000001
- `triton_poi_fused_mul_silu_slice_1`: 55.634
- `triton_poi_fused_3`: 50.002
- `triton_poi_fused_1`: 47.698
- `_combine_sampled_and_draft_tokens_kernel`: 39.308
- `_prepare_pos_seq_lens_kernel`: 21.514000000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 19.398
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.07

## mimo-v26 prefill64k

- `triton_poi_fused_2`: 16614.466
- `_apply_write_kernel`: 897.4050000000001
- `triton_poi_fused_mul_silu_slice_1`: 502.701
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 394.892
- `triton_poi_fused_3`: 362.85900000000004
- `triton_poi_fused_1`: 357.804
- `_prepare_prefill_inputs_kernel`: 234.406
- `_combine_sampled_and_draft_tokens_kernel`: 39.81
- `_prepare_pos_seq_lens_kernel`: 37.857
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.48
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 16.193
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.744
