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

## qwen-38-bf16 decode128k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 72303.655
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 53943.459
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 39019.236
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27086.916
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 22312.154000000002
- `_qsa_prepare_kernel`: 16989.143999999997
- `_qsa_merge_splitk_kernel`: 11579.828000000001
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5464.6359999999995
- `_gumbel_sample_kernel`: 2943.784
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2497.185
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2422.146
- `_ple_conv_kernel`: 2242.694

## qwen-38-bf16 decode128k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 207571.329
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 42767.264
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 31168.228000000003
- `_qsa_prepare_kernel`: 20365.050999999996
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 18667.269999999993
- `_qsa_merge_splitk_kernel`: 12337.626
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3797.9239999999995
- `_gumbel_sample_kernel`: 3481.5589999999997
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2691.913
- `_ple_conv_kernel`: 2327.048
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2290.163
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 2014.9550000000002

## qwen-38-bf16 decode16k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 74948.328
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 56008.996
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 40473.798
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28092.067
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 23169.475000000002
- `_qsa_prepare_kernel`: 17684.990999999998
- `_qsa_merge_splitk_kernel`: 12021.305
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5683.377
- `_gumbel_sample_kernel`: 3061.703
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2606.237
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2557.030999999999
- `_ple_conv_kernel`: 2360.2860000000005

## qwen-38-bf16 decode16k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 190396.70300000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 38691.289000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28291.083
- `_qsa_prepare_kernel`: 19913.676
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 16941.983999999997
- `_qsa_merge_splitk_kernel`: 10801.230999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3455.2
- `_gumbel_sample_kernel`: 3059.5700000000006
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2433.5139999999997
- `_ple_conv_kernel`: 2116.334
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2079.5609999999997
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1832.7900000000002

## qwen-38-bf16 decode1k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 74037.014
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 54982.28399999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 40477.76299999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27780.779000000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 22916.084000000003
- `_qsa_prepare_kernel`: 17278.772999999997
- `_qsa_merge_splitk_kernel`: 11872.395999999999
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5644.931
- `_gumbel_sample_kernel`: 3224.5430000000006
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2576.759
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2509.7989999999995
- `_ple_conv_kernel`: 2342.7059999999997

## qwen-38-bf16 decode1k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 190654.89200000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 38836.090000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28303.484
- `_qsa_prepare_kernel`: 20074.179999999997
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 17023.671
- `_qsa_merge_splitk_kernel`: 10726.045
- `_gumbel_sample_kernel`: 3554.8340000000003
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3446.493
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2462.57
- `_ple_conv_kernel`: 2125.248
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2075.5579999999995
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1832.1280000000002

## qwen-38-bf16 decode64k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 73505.97400000002
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 54872.184
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 39737.981
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27551.624000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 22699.486
- `_qsa_prepare_kernel`: 17241.464000000004
- `_qsa_merge_splitk_kernel`: 11759.986
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5540.885
- `_gumbel_sample_kernel`: 3006.5
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2529.234
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2438.021
- `_ple_conv_kernel`: 2324.67

## qwen-38-bf16 decode64k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 189359.46699999998
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 38686.448000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28289.597999999998
- `_qsa_prepare_kernel`: 20010.374000000003
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 16917.631999999994
- `_qsa_merge_splitk_kernel`: 11205.038
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3446.523
- `_gumbel_sample_kernel`: 2907.131
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2460.139
- `_ple_conv_kernel`: 2122.272
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2080.709
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1832.0369999999998

## qwen-38-bf16 prefill128k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 59711.10900000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 31748.553000000004
- `_ple_conv_kernel`: 6659.734
- `_qsa_prepare_kernel`: 6201.898000000001
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5301.668
- `_lookup_ple_embedding_from_pinned_kernel`: 4779.378
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 3742.984
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2901.181
- `_apply_write_kernel`: 1749.7579999999998
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1394.4300000000003
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1282.8880000000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 873.1380000000001

## qwen-38-bf16 prefill16k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 59631.240999999995
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 31751.472
- `_ple_conv_kernel`: 6613.648000000001
- `_qsa_prepare_kernel`: 6205.311000000001
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5298.463999999999
- `_lookup_ple_embedding_from_pinned_kernel`: 5096.262999999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 3737.8789999999995
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2894.36
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1392.9820000000002
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1283.2130000000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 877.5050000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIt`: 816.665

## qwen-38-bf16 prefill1k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 10791.140000000001
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5206.404
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 3638.796
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2454.1979999999994
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1756.7780000000002
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1287.8460000000005
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1182.15
- `void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIt`: 1095.6540000000002
- `_qsa_prepare_kernel`: 1086.494
- `_ple_conv_kernel`: 902.5559999999999
- `_qsa_merge_splitk_kernel`: 795.382
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 766.4279999999999

## qwen-38-bf16 prefill64k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 59710.845
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 31781.471999999998
- `_ple_conv_kernel`: 6660.465
- `_qsa_prepare_kernel`: 6204.092
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5297.889999999999
- `_lookup_ple_embedding_from_pinned_kernel`: 5210.047
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 3745.143000000001
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2894.133999999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1395.506
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1286.162
- `_apply_write_kernel`: 977.7370000000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 871.0989999999997

## glm-53 decode128k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 34882.51
- `_fwht_quant_kernel`: 23711.379999999997
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 22507.697
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 15159.821000000004
- `sparse_mla_index_remap_kernel`: 14322.464000000002
- `Kernel`: 8974.795
- `_kpool_decode_update_batched_kernel`: 7660.712
- `memcpy32_post`: 6176.91
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 5371.384000000002
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5368.148000000001
- `_expand_pools_and_append_tail_kernel`: 5260.076000000002
- `triton_poi_fused_mul_unsqueeze_0`: 4941.095

## glm-53 decode128k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 36079.342
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26920.293999999994
- `_fwht_quant_kernel`: 23295.018999999997
- `sparse_mla_index_remap_kernel`: 14111.27
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8764.91
- `Kernel`: 8495.284
- `_kpool_decode_update_batched_kernel`: 8186.469000000001
- `_expand_pools_and_append_tail_kernel`: 6288.1179999999995
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5203.271000000002
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4950.736
- `triton_poi_fused_mul_unsqueeze_0`: 4593.9180000000015
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 3075.7380000000003

## glm-53 decode16k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 32324.228
- `_fwht_quant_kernel`: 21995.400999999998
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20880.302
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 14075.171
- `sparse_mla_index_remap_kernel`: 13115.252999999999
- `Kernel`: 8329.009000000002
- `_kpool_decode_update_batched_kernel`: 7125.961000000001
- `memcpy32_post`: 5804.324000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4994.188000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4979.306000000001
- `_expand_pools_and_append_tail_kernel`: 4902.663000000001
- `triton_poi_fused_mul_unsqueeze_0`: 4585.354

## glm-53 decode16k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 36429.652
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27452.763000000003
- `_fwht_quant_kernel`: 23704.839999999997
- `sparse_mla_index_remap_kernel`: 14083.965999999997
- `_kpool_decode_update_batched_kernel`: 9455.652000000002
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8938.016999999998
- `Kernel`: 8668.79
- `_expand_pools_and_append_tail_kernel`: 6361.059000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5287.102000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 5050.7570000000005
- `triton_poi_fused_mul_unsqueeze_0`: 4671.1
- `_gumbel_sample_kernel`: 3635.791

## glm-53 decode1k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 31927.897999999994
- `_fwht_quant_kernel`: 21819.087
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20566.263000000003
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 13869.553999999998
- `sparse_mla_index_remap_kernel`: 12977.011000000002
- `Kernel`: 8230.227
- `_kpool_decode_update_batched_kernel`: 7013.180999999999
- `memcpy32_post`: 5646.690999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4930.329
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4911.1539999999995
- `_expand_pools_and_append_tail_kernel`: 4801.21
- `triton_poi_fused_mul_unsqueeze_0`: 4519.44

## glm-53 decode1k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 36451.901999999995
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27463.395999999997
- `_fwht_quant_kernel`: 23749.746
- `sparse_mla_index_remap_kernel`: 14127.177
- `_kpool_decode_update_batched_kernel`: 10536.628999999999
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8951.49
- `Kernel`: 8680.269999999999
- `_expand_pools_and_append_tail_kernel`: 6350.219000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5301.048000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 5069.459000000001
- `triton_poi_fused_mul_unsqueeze_0`: 4680.657
- `_gumbel_sample_kernel`: 3228.3590000000004

## glm-53 decode64k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 32423.514999999996
- `_fwht_quant_kernel`: 22044.567
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20899.058999999997
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 14092.563999999998
- `sparse_mla_index_remap_kernel`: 13239.708999999999
- `Kernel`: 8336.896999999999
- `_kpool_decode_update_batched_kernel`: 7110.597000000001
- `memcpy32_post`: 5817.446000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4984.488000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4984.219000000002
- `_expand_pools_and_append_tail_kernel`: 4906.4529999999995
- `triton_poi_fused_mul_unsqueeze_0`: 4591.136

## glm-53 decode64k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 35751.028000000006
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26928.841
- `_fwht_quant_kernel`: 23294.933
- `sparse_mla_index_remap_kernel`: 13945.267999999998
- `_kpool_decode_update_batched_kernel`: 10290.555
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8764.046999999999
- `Kernel`: 8497.274
- `_expand_pools_and_append_tail_kernel`: 6245.843000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5197.556
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4948.424000000002
- `triton_poi_fused_mul_unsqueeze_0`: 4591.312000000002
- `_gumbel_sample_kernel`: 3114.3469999999998

## glm-53 prefill128k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 45438.553
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 44553.492000000006
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 21848.719000000005
- `sparse_mla_index_remap_kernel`: 15269.977
- `_expand_pools_and_append_tail_kernel`: 15068.597
- `_fwht_quant_kernel`: 13533.455000000002
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 10475.502999999999
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 7462.518000000001
- `Kernel`: 5175.66
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 3945.867
- `_gather_initial_states_kernel`: 2818.148
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::Mea`: 2446.959

## glm-53 prefill16k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 45337.26400000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 44524.693999999996
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 21459.482
- `sparse_mla_index_remap_kernel`: 15175.743999999999
- `_expand_pools_and_append_tail_kernel`: 14960.969999999998
- `_fwht_quant_kernel`: 13430.713999999998
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 10538.006
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 7456.891999999999
- `Kernel`: 5117.554
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 3923.1099999999997
- `_gather_initial_states_kernel`: 2603.79
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::Mea`: 2447.1359999999995

## glm-53 prefill1k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 8581.762
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 4217.464
- `_gather_initial_states_kernel`: 2558.136
- `_fwht_quant_kernel`: 2272.2980000000002
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2139.694
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1501.0259999999998
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 1288.124
- `_scatter_states_kernel`: 751.0200000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::maske`: 744.0960000000001
- `Kernel`: 739.5459999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 593.582
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 554.54

## glm-53 prefill64k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 45410.569
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 44523.517
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 21476.977000000003
- `sparse_mla_index_remap_kernel`: 15232.616000000002
- `_expand_pools_and_append_tail_kernel`: 14969.209
- `_fwht_quant_kernel`: 13522.496000000001
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 10568.809000000001
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 7462.923000000001
- `Kernel`: 5137.1720000000005
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 3929.541
- `_gather_initial_states_kernel`: 2822.304
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::Mea`: 2454.4379999999996
