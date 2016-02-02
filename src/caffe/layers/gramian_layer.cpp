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
}

template <typename Dtype>
void GramianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape(2);
    top_shape[0] = M_;
    top_shape[1] = K_*K_;
    top[0]->Reshape(top_shape);

}

template <typename Dtype>
void GramianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    for (int batch=0; batch < M_; ++batch) {
        const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(batch);
        Dtype* mutable_top_data = top[0]->mutable_cpu_data() + top[0]->offset(batch);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, N_, (Dtype)1.,
          bottom_data, bottom_data, (Dtype)0., mutable_top_data);
    }
}

template <typename Dtype>
void GramianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    for (int batch=0; batch < M_; ++batch) {
        Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(batch);
        const Dtype* bottom_diff = bottom[0]->cpu_diff() + bottom[0]->offset(batch);
        const Dtype* top_diff = top[0]->mutable_cpu_diff() + top[0]->offset(batch);
        // Input gradient
        if (propagate_down[0]) {
            caffe_scal(bottom[0]->count(1), (Dtype)2., mutable_bottom_diff);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, K_, (Dtype)1.,
              top_diff, bottom_diff, (Dtype)1., mutable_bottom_diff);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(GramianLayer);
#endif

INSTANTIATE_CLASS(GramianLayer);
REGISTER_LAYER_CLASS(Gramian);

}  // namespace caffe
