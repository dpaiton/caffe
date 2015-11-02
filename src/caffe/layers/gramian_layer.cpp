#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    M_ = bottom[0]->shape(0); // batch size
    K_ = bottom[0]->shape(1); // num elements
    N_ = bottom[0]->count(2); // num pixels
    normalize_scale_ = 1.0;
    if (this->layer_param_.gramian_param().normalize_output()) {
        normalize_scale_ = 1.0 / pow(2.0 * K_ * N_, 2.0);
    }
}

template <typename Dtype>
void GramianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape(2);
    top_shape[0] = M_;
    top_shape[1] = K_ * K_;
    top[0]->Reshape(top_shape);

    vector<int> bot_temp_shape(2);
    bot_temp_shape[0] = K_;
    bot_temp_shape[1] = N_;
    bot_temp_1_.Reshape(bot_temp_shape);
    bot_temp_2_.Reshape(bot_temp_shape);

}

template <typename Dtype>
void GramianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    for (int batch=0; batch < M_; ++batch) {
        const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(batch);
        Dtype* mutable_top_data = top[0]->mutable_cpu_data() + top[0]->offset(batch);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, N_, normalize_scale_,
          bottom_data, bottom_data, (Dtype)0., mutable_top_data);
    }
}

template <typename Dtype>
void GramianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    for (int batch=0; batch < M_; ++batch) {
        Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(batch);
        const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(batch);
        const Dtype* top_diff = top[0]->cpu_diff() + top[0]->offset(batch);
        // Input gradient
        if (propagate_down[0]) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, K_, normalize_scale_,
              top_diff, bottom_data, (Dtype)0., bot_temp_1_.mutable_cpu_diff());
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, K_, normalize_scale_,
              top_diff, bottom_data, (Dtype)0., bot_temp_2_.mutable_cpu_diff());
            caffe_add(K_*N_, bot_temp_1_.cpu_diff(), bot_temp_2_.cpu_diff(), mutable_bottom_diff);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(GramianLayer);
#endif

INSTANTIATE_CLASS(GramianLayer);
REGISTER_LAYER_CLASS(Gramian);

}  // namespace caffe
