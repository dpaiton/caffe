#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    for (int batch=0; batch < M_; ++batch) {
        const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(batch);
        Dtype* mutable_top_data = top[0]->mutable_gpu_data() + top[0]->offset(batch);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, N_, normalize_scale_,
          bottom_data, bottom_data, (Dtype)0., mutable_top_data);
    }
}

template <typename Dtype>
void GramianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    for (int batch=0; batch < M_; ++batch) {
        Dtype* mutable_bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(batch);
        const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(batch);
        const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(batch);
        // Input gradient
        if (propagate_down[0]) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, K_, normalize_scale_,
              top_diff, bottom_data, (Dtype)0., bot_temp_1_.mutable_gpu_diff());
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, K_, normalize_scale_,
              top_diff, bottom_data, (Dtype)0., bot_temp_2_.mutable_gpu_diff());
            caffe_gpu_add(K_*N_, bot_temp_1_.gpu_diff(), bot_temp_2_.gpu_diff(), mutable_bottom_diff);
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(GramianLayer);

}  // namespace caffe
