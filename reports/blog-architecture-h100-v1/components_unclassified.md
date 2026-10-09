# Largest kernels left in `other_elementwise` (µs summed over the averaged steps, all ranks)

## glm-53 decode1k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 29672.238000000012
- `_fwht_quant_kernel`: 20286.10099999998
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 19191.187
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 13042.972000000003
- `sparse_mla_index_remap_kernel`: 12566.442000000001
- `Kernel`: 7946.988999999998
- `_kpool_decode_update_batched_kernel`: 6600.338000000002
- `memcpy32_post`: 5353.698999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4630.441999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4564.715
- `_expand_pools_and_append_tail_kernel`: 4485.864
- `triton_poi_fused_mul_unsqueeze_0`: 4277.167

## glm-53 decode1k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 33618.752000000015
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 24755.316000000006
- `_fwht_quant_kernel`: 22117.518999999986
- `sparse_mla_index_remap_kernel`: 13425.628000000002
- `_kpool_decode_update_batched_kernel`: 9780.986999999997
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8880.137
- `Kernel`: 8389.896000000008
- `_expand_pools_and_append_tail_kernel`: 5752.817000000003
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4870.819
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4582.634000000003
- `triton_poi_fused_mul_unsqueeze_0`: 4232.578
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 2860.0129999999976

## glm-53 decode64k_B1

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 29485.51299999998
- `_fwht_quant_kernel`: 20090.130999999983
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 19087.018000000022
- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>`: 12981.081000000007
- `sparse_mla_index_remap_kernel`: 12712.100000000004
- `Kernel`: 7897.7219999999925
- `_kpool_decode_update_batched_kernel`: 6565.0729999999985
- `memcpy32_post`: 5341.245
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4604.224999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4533.955000000003
- `_expand_pools_and_append_tail_kernel`: 4461.645999999997
- `triton_poi_fused_mul_unsqueeze_0`: 4252.630999999999

## glm-53 decode64k_B8

- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 33614.70700000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 24807.902999999984
- `_fwht_quant_kernel`: 22145.89699999999
- `sparse_mla_index_remap_kernel`: 13715.193000000001
- `_kpool_decode_update_batched_kernel`: 9773.893999999998
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 8877.556000000002
- `Kernel`: 8374.350999999999
- `_expand_pools_and_append_tail_kernel`: 5754.796999999996
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4875.982000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, `: 4588.666000000003
- `triton_poi_fused_mul_unsqueeze_0`: 4234.615000000002
- `_gumbel_sample_kernel`: 3010.83

## glm-53 prefill16k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 45393.012
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 44695.698
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 21494.536
- `sparse_mla_index_remap_kernel`: 15358.022
- `_expand_pools_and_append_tail_kernel`: 14972.143999999998
- `_fwht_quant_kernel`: 13446.602000000003
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 10658.708
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 7454.992000000001
- `Kernel`: 5124.925999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 3932.0460000000003
- `_gather_initial_states_kernel`: 2560.962
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::Mea`: 2444.5820000000003

## glm-53 prefill64k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 45514.257
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 44712.305
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 21540.032000000003
- `sparse_mla_index_remap_kernel`: 15532.607999999997
- `_expand_pools_and_append_tail_kernel`: 14974.190000000002
- `_fwht_quant_kernel`: 13576.364
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 10672.344
- `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, `: 7458.806
- `Kernel`: 5136.191999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 3933.3729999999996
- `_gather_initial_states_kernel`: 2804.894
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::Mea`: 2446.651

## mimo-v26 decode1k_B1

- `triton_poi_fused_0`: 18727.234
- `memcpy32_post`: 12282.213999999993
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 2021.700999999999
- `_gumbel_sample_kernel`: 1800.1609999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1139.15
- `_post_update_kernel`: 971.7050000000006
- `triton_poi_fused_1`: 952.6379999999996
- `_combine_sampled_and_draft_tokens_kernel`: 744.5820000000004
- `_get_num_sampled_and_rejected_kernel`: 736.1140000000008
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 527.0600000000002
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 486.4579999999992
- `_prepare_pos_seq_lens_kernel`: 368.4299999999993

## mimo-v26 decode1k_B8

- `triton_poi_fused_0`: 308641.43799999973
- `memcpy32_post`: 185724.13999999975
- `_gumbel_sample_kernel`: 26174.95300000003
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 19346.193999999963
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 18678.199999999964
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 16211.815999999892
- `triton_poi_fused_1`: 14808.828000000092
- `_post_update_kernel`: 14220.91800000009
- `_combine_sampled_and_draft_tokens_kernel`: 13252.773000000036
- `_get_num_sampled_and_rejected_kernel`: 12376.248999999996
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 7894.231999999979
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 7075.751000000075

## mimo-v26 decode64k_B1

- `triton_poi_fused_0`: 18769.786999999997
- `memcpy32_post`: 12420.504000000006
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 2056.8519999999985
- `_gumbel_sample_kernel`: 1684.4729999999997
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1160.4780000000003
- `_post_update_kernel`: 992.1710000000002
- `triton_poi_fused_1`: 944.7009999999999
- `_combine_sampled_and_draft_tokens_kernel`: 755.7869999999991
- `_get_num_sampled_and_rejected_kernel`: 755.2149999999996
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 536.3439999999998
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 493.89400000000023
- `_prepare_pos_seq_lens_kernel`: 374.8299999999992

## mimo-v26 decode64k_B8

- `triton_poi_fused_0`: 281786.6709999999
- `memcpy32_post`: 168231.63799999957
- `_gumbel_sample_kernel`: 24426.924000000017
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 17580.95599999991
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 16980.84499999995
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 14738.641999999845
- `triton_poi_fused_1`: 13540.08100000006
- `_post_update_kernel`: 12901.874000000087
- `_combine_sampled_and_draft_tokens_kernel`: 11965.652000000075
- `_get_num_sampled_and_rejected_kernel`: 11243.561999999978
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 7171.713999999972
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 6434.81300000007

## mimo-v26 prefill16k

- `triton_poi_fused_2`: 16587.675999999996
- `_apply_write_kernel`: 619.1040000000002
- `triton_poi_fused_mul_silu_slice_1`: 500.7439999999999
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 406.65200000000004
- `triton_poi_fused_3`: 357.17600000000004
- `triton_poi_fused_1`: 350.1940000000001
- `_prepare_prefill_inputs_kernel`: 214.81399999999996
- `_combine_sampled_and_draft_tokens_kernel`: 37.63599999999999
- `_prepare_pos_seq_lens_kernel`: 36.422
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 15.938000000000006
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 15.684000000000005
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.362000000000005

## mimo-v26 prefill64k

- `triton_poi_fused_2`: 16688.936
- `_apply_write_kernel`: 911.2549999999999
- `triton_poi_fused_mul_silu_slice_1`: 505.28300000000013
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 397.5400000000001
- `triton_poi_fused_3`: 363.323
- `triton_poi_fused_1`: 356.94500000000005
- `_prepare_prefill_inputs_kernel`: 232.60399999999998
- `_combine_sampled_and_draft_tokens_kernel`: 39.376000000000005
- `_prepare_pos_seq_lens_kernel`: 37.806
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.564
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 16.436
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.729999999999999

## v4-0731 decode1k_B1

- `Kernel`: 123809.565
- `memcpy32_post`: 29448.259999999984
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20958.82600000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9510.638999999997
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 2924.8290000000006
- `_gumbel_sample_kernel`: 2719.819999999999
- `hc_head_fuse_tilelang_kernel`: 2662.9080000000004
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2579.2880000000005
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2553.0109999999986
- `_compute_swa_indices_and_lens_kernel`: 1705.8839999999996
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1684.6740000000011
- `_post_update_kernel`: 1437.836000000001

## v4-0731 decode1k_B8

- `Kernel`: 125752.19100000002
- `memcpy32_post`: 28494.689000000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 26274.027000000002
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9175.838
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3280.3419999999965
- `_gumbel_sample_kernel`: 3209.1000000000004
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2988.4260000000013
- `hc_head_fuse_tilelang_kernel`: 2667.1409999999983
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1792.907999999999
- `_compute_swa_indices_and_lens_kernel`: 1760.4309999999978
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1636.2769999999987
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1619.297999999998

## v4-0731 decode64k_B1

- `Kernel`: 123097.31200000006
- `memcpy32_post`: 29372.369000000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 20779.715000000022
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 9401.735999999995
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 2895.687
- `_gumbel_sample_kernel`: 2810.7879999999986
- `hc_head_fuse_tilelang_kernel`: 2643.510999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 2536.1050000000027
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2514.5250000000024
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1663.5289999999993
- `_compute_swa_indices_and_lens_kernel`: 1653.0060000000026
- `_post_update_kernel`: 1408.5289999999995

## v4-0731 decode64k_B8

- `Kernel`: 125680.09899999999
- `memcpy32_post`: 28248.36400000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 25755.115999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 8987.244999999999
- `_gumbel_sample_kernel`: 3364.5650000000005
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 3190.5769999999984
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 2931.9669999999987
- `hc_head_fuse_tilelang_kernel`: 2615.459000000001
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1760.3399999999976
- `_compute_swa_indices_and_lens_kernel`: 1756.34
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1597.5410000000006
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1586.5519999999972

## v4-0731 prefill16k

- `Kernel`: 116945.89200000005
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 46734.342000000004
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4933.662
- `hc_head_fuse_tilelang_kernel`: 3822.56
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1178.2700000000004
- `_apply_write_kernel`: 643.1859999999999
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 443.0400000000001
- `_compute_swa_indices_and_lens_kernel`: 410.33400000000006
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 335.95599999999996
- `_prepare_prefill_inputs_kernel`: 218.642
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 116.29400000000004
- `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::arra`: 48.897999999999996

## v4-0731 prefill64k

- `Kernel`: 118042.096
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 46711.094000000005
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 5691.432000000002
- `hc_head_fuse_tilelang_kernel`: 3820.6620000000003
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1184.2140000000002
- `_apply_write_kernel`: 988.5760000000001
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 454.547
- `_compute_swa_indices_and_lens_kernel`: 421.846
- `void vllm::moe::dsv4HashTopkSoftplusSqrt<int, int>(float const*, float*, int*, int, int, f`: 341.62100000000004
- `_prepare_prefill_inputs_kernel`: 239.69299999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launc`: 115.749
- `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::arra`: 49.63400000000001

## v41 decode1k_B1

- `Kernel`: 31458.905999999988
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 13784.378000000002
- `memcpy32_post`: 11556.265000000005
- `_compute_swa_indices_and_lens_kernel`: 3961.015000000001
- `_candidate_flags_kernel`: 3299.640000000002
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 2592.4520000000007
- `_gumbel_sample_kernel`: 2343.285
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 2293.0819999999976
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 2137.9500000000003
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 1786.473
- `_hash_ids_kernel`: 1240.1630000000005
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1211.4319999999998

## v41 decode1k_B8

- `Kernel`: 31007.050999999992
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 17200.079
- `memcpy32_post`: 10117.852999999992
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 4348.531999999998
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4201.848000000001
- `_compute_swa_indices_and_lens_kernel`: 3703.5489999999995
- `_candidate_flags_kernel`: 3246.3490000000006
- `_gumbel_sample_kernel`: 2318.8740000000007
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 2172.1339999999996
- `_hash_ids_kernel`: 1248.952
- `_combine_sampled_and_draft_tokens_kernel`: 1200.064
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1192.8810000000005

## v41 decode64k_B1

- `Kernel`: 29425.323000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 12890.122
- `memcpy32_post`: 10854.710000000001
- `_candidate_flags_kernel`: 4493.601999999999
- `_compute_swa_indices_and_lens_kernel`: 3657.6600000000017
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 2409.9249999999984
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 2136.9790000000003
- `_gumbel_sample_kernel`: 1952.8919999999994
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1925.002999999999
- `void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<int>, std::array<char`: 1666.15
- `_hash_ids_kernel`: 1151.1809999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1131.6130000000012

## v41 decode64k_B8

- `Kernel`: 35809.698000000026
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 19466.54900000001
- `memcpy32_post`: 11446.619
- `_candidate_flags_kernel`: 5594.485999999996
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 4923.531999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 4792.264999999999
- `_compute_swa_indices_and_lens_kernel`: 4254.406999999998
- `_gumbel_sample_kernel`: 2511.2640000000006
- `void at::native::vectorized_elementwise_kernel<8, at::native::compare_scalar_kernel<long>(`: 2493.16
- `_hash_ids_kernel`: 1431.8559999999998
- `_combine_sampled_and_draft_tokens_kernel`: 1381.9250000000013
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1367.0030000000002

## v41 prefill16k

- `Kernel`: 46073.14800000001
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 33320.238
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28930.814
- `_hc_head_reduce_store_kernel`: 2304.370000000001
- `memcpy128`: 2002.2920000000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1661.958
- `_compute_swa_indices_and_lens_kernel`: 1656.326
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1126.766
- `_store_candidates_kernel`: 1056.232
- `_apply_write_kernel`: 677.1020000000001
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 524.4979999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 476.23400000000004

## v41 prefill64k

- `Kernel`: 47932.352000000006
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 30810.812999999995
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28917.886
- `_hc_head_reduce_store_kernel`: 2311.439
- `memcpy128`: 2002.254
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 1668.8709999999996
- `_compute_swa_indices_and_lens_kernel`: 1652.3049999999998
- `_store_candidates_kernel`: 1209.5499999999997
- `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int,`: 1125.987
- `_apply_write_kernel`: 990.15
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 522.5600000000001
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 507.13200000000006
