# Largest kernels left in `other_elementwise` (µs summed over the averaged steps, all ranks)

## mimo-v26 decode1k_B1

- `triton_poi_fused_0`: 461113.4529999993
- `memcpy32_post`: 304543.3770000008
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 50352.67200000011
- `_gumbel_sample_kernel`: 42898.028999999755
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 27877.546999999522
- `_post_update_kernel`: 24104.049000000505
- `triton_poi_fused_1`: 22774.248000000334
- `_combine_sampled_and_draft_tokens_kernel`: 18235.53200000081
- `_get_num_sampled_and_rejected_kernel`: 17760.048000000264
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 13342.563999999902
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 12981.217000000235
- `_prepare_pos_seq_lens_kernel`: 9044.016000000778

## mimo-v26 decode1k_B8

- `triton_poi_fused_0`: 30973.399000000012
- `memcpy32_post`: 18522.47000000001
- `_gumbel_sample_kernel`: 3031.3269999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1932.2820000000013
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1869.1670000000036
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1646.4700000000018
- `triton_poi_fused_1`: 1491.7889999999973
- `_post_update_kernel`: 1382.4919999999981
- `_combine_sampled_and_draft_tokens_kernel`: 1273.0049999999987
- `_get_num_sampled_and_rejected_kernel`: 1204.2540000000017
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 793.0869999999998
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 693.3369999999984

## mimo-v26 decode64k_B1

- `triton_poi_fused_0`: 450873.6249999992
- `memcpy32_post`: 298083.4159999994
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 49280.147000000274
- `_gumbel_sample_kernel`: 41789.78799999984
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 27311.97899999949
- `_post_update_kernel`: 23568.351000000453
- `triton_poi_fused_1`: 22417.258000000274
- `_combine_sampled_and_draft_tokens_kernel`: 17868.659000000727
- `_get_num_sampled_and_rejected_kernel`: 17390.81200000031
- `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gath`: 13097.683999999896
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 12719.384000000195
- `_prepare_pos_seq_lens_kernel`: 8862.471000000787

## mimo-v26 decode64k_B8

- `triton_poi_fused_0`: 30438.09899999997
- `memcpy32_post`: 18099.58100000002
- `_gumbel_sample_kernel`: 2446.3450000000034
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1895.725000000001
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 1828.143000000003
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 1611.1360000000025
- `triton_poi_fused_1`: 1470.8279999999988
- `_post_update_kernel`: 1357.3779999999997
- `_combine_sampled_and_draft_tokens_kernel`: 1232.9569999999994
- `_get_num_sampled_and_rejected_kernel`: 1178.4890000000012
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 776.6149999999997
- `_apply_write_kernel`: 703.1350000000001

## mimo-v26 prefill16k

- `triton_poi_fused_2`: 16629.107999999997
- `_apply_write_kernel`: 669.826
- `triton_poi_fused_mul_silu_slice_1`: 505.02599999999995
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 410.242
- `triton_poi_fused_3`: 357.376
- `triton_poi_fused_1`: 353.8560000000001
- `_prepare_prefill_inputs_kernel`: 215.746
- `_combine_sampled_and_draft_tokens_kernel`: 37.75999999999999
- `_prepare_pos_seq_lens_kernel`: 36.48
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.062000000000005
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 16.000000000000007
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.552000000000007

## mimo-v26 prefill64k

- `triton_poi_fused_2`: 16673.664
- `_apply_write_kernel`: 919.04
- `triton_poi_fused_mul_silu_slice_1`: 506.17799999999994
- `void vllm::vocab_embedding::vocab_parallel_embedding_kernel<int, uint4>(uint4*, int const*`: 398.59000000000003
- `triton_poi_fused_1`: 358.879
- `triton_poi_fused_3`: 358.08099999999996
- `_prepare_prefill_inputs_kernel`: 231.069
- `_combine_sampled_and_draft_tokens_kernel`: 39.23
- `_prepare_pos_seq_lens_kernel`: 38.179
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array`: 16.608000000000008
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<bool>, std::arra`: 16.224000000000007
- `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out`: 15.777000000000006

## qwen-38-bf16 decode1k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 64980.85599999999
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 49942.359000000026
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 36496.557999999975
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 24617.81599999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 20015.817999999985
- `_qsa_prepare_kernel`: 15446.37
- `_qsa_merge_splitk_kernel`: 10486.502
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5028.416000000004
- `_gumbel_sample_kernel`: 2630.6449999999986
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2304.1250000000005
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2105.888
- `_ple_conv_kernel`: 2044.3699999999985

## qwen-38-bf16 decode1k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 180522.568
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 38779.076000000015
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 28002.625999999986
- `_qsa_prepare_kernel`: 19879.088999999985
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 17057.372000000003
- `_qsa_merge_splitk_kernel`: 10834.086999999998
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3504.718000000007
- `_gumbel_sample_kernel`: 3153.9480000000026
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2428.9659999999985
- `_ple_conv_kernel`: 2185.721999999999
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2125.986999999999
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1830.929000000003

## qwen-38-bf16 decode64k_B1

- `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1`: 184489.93099999998
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 141603.2170000002
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 103611.99400000004
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 69878.65999999996
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 56959.743000000046
- `_qsa_prepare_kernel`: 43740.92599999997
- `_qsa_merge_splitk_kernel`: 30151.544000000016
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 14257.971999999949
- `_gumbel_sample_kernel`: 7408.059
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 6539.789000000013
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 6005.54200000001
- `_ple_conv_kernel`: 5776.5899999999965

## qwen-38-bf16 decode64k_B8

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 182475.6660000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 37934.16000000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 27541.398
- `_qsa_prepare_kernel`: 19626.85100000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 16671.925000000007
- `_qsa_merge_splitk_kernel`: 11036.858
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 3405.293000000005
- `_gumbel_sample_kernel`: 2758.1980000000003
- `void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals`: 2432.057999999999
- `_ple_conv_kernel`: 2135.6979999999985
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 2085.804999999998
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<f`: 1807.150000000003

## qwen-38-bf16 prefill16k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 59677.31900000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 32024.586
- `_ple_conv_kernel`: 6661.3820000000005
- `_qsa_prepare_kernel`: 6236.0070000000005
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5254.778999999999
- `_lookup_ple_embedding_from_pinned_kernel`: 5184.099
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 3728.0760000000005
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 3002.9919999999997
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1395.9640000000004
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1355.1770000000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 882.5919999999999
- `void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIt`: 813.2810000000002

## qwen-38-bf16 prefill64k

- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`: 59702.28599999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1`: 32031.562
- `_ple_conv_kernel`: 6672.897
- `_qsa_prepare_kernel`: 6250.859000000001
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:`: 5258.156999999998
- `_lookup_ple_embedding_from_pinned_kernel`: 5105.666999999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, `: 3724.3770000000004
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso`: 2992.039999999999
- `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens`: 1393.458
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native:`: 1354.5349999999999
- `_apply_write_kernel`: 988.2870000000001
- `void at::native::vectorized_elementwise_kernel<8, at::native::bitwise_not_kernel_cuda(at::`: 879.0029999999999
