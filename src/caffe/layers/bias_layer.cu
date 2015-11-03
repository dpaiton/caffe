#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[0]->gpu_data();
  Dtype* mutable_top_data = top[0]->mutable_gpu_data();
  caffe_gpu_add(bottom[0]->count(), bias_data, bottom_data, mutable_top_data);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) { // Gradient with respect to bias
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* mutable_bias_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_copy(this->blobs_[0]->count(), top_diff, mutable_bias_diff);
  }
  if (propagate_down[0]) { // Gradient with respect to bottom data
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* mutable_bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_copy(bottom[0]->count(), top_diff, mutable_bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
