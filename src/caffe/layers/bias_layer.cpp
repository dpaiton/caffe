#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // intiialize and fill the bias term
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(bottom[0]->shape()));
  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
      this->layer_param_.inner_product_param().bias_filler()));
  bias_filler->Fill(this->blobs_[0].get());
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bias_data = this->blobs_[0]->cpu_data();
  Dtype* mutable_top_data = top[0]->mutable_cpu_data();
  caffe_add(bottom[0]->count(), bias_data, bottom_data, mutable_top_data);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) { // Gradient with respect to bias
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* mutable_bias_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_copy(this->blobs_[0]->count(), top_diff, mutable_bias_diff);
  }
  if (propagate_down[0]) { // Gradient with respect to bottom data
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, mutable_bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasLayer);
#endif

INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe
