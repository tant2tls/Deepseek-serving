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
