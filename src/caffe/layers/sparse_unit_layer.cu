#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* weights    = this->blobs_[0]->gpu_data(); // phi    :: N_xK_
  const Dtype* a_past;                                   // a(t-1) :: M_xK_
  Dtype* mutable_top_data = top[0]->mutable_gpu_data();  // output :: M_xK_

  if (bottom.size() == 1) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        bottom[0]->gpu_data(), weights, (Dtype)0., previous_activity_.mutable_gpu_data());
    a_past = previous_activity_.gpu_data();
  } else {
    a_past = bottom[1]->gpu_data();
  }

  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->gpu_data(); // bias   :: 1xN_
    // Replicate bias vector into M_xN_ matrix, store in temp_1_
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        batch_multiplier_.gpu_data(), bias, (Dtype)0., temp_1_.mutable_gpu_data());
    // Subtract bias values from input
    caffe_gpu_sub(bottom[0]->count(), bottom[0]->gpu_data(), temp_1_.gpu_data(),
        biased_input_.mutable_gpu_data());
  } else {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), biased_input_.mutable_gpu_data());
  }

  // competition_matrix_ = phi^Tphi
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, N_, (Dtype)1.,
      weights, weights, (Dtype)0., competition_matrix_.mutable_gpu_data());

  // top = sgn(a[t-1])
  caffe_gpu_sign(top[0]->count(), a_past, mutable_top_data);

  // top = a[t-1] G + lambda_ sgn(a[t-1])
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
      a_past, competition_matrix_.gpu_data(), lambda_, mutable_top_data);

  // top = (s-b) w - (a[t-1] G + lambda_ sgn(a[t-1]))
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      biased_input_.gpu_data(), weights, (Dtype)-1., mutable_top_data);

  // a[t-1] + eta_ ((s-b) w - a[t-1] G - lambda_ sgn(a[t-1]))
  caffe_gpu_axpby(top[0]->count(), (Dtype)1., a_past, eta_, mutable_top_data);
}

template <typename Dtype>
void SparseUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* weights = this->blobs_[0]->gpu_data();
  
  if (this->param_propagate_down_[0]) { // Weight gradient
    const Dtype* a_past;
    if (bottom.size() == 1) {
      a_past = previous_activity_.gpu_data();
    } else {
      a_past = bottom[1]->gpu_data();
    }
    Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        a_past, weights, (Dtype)0., temp_1_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
        a_past, top[0]->gpu_diff(), (Dtype)0., temp_2_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, -eta_,
        temp_1_.gpu_data(), top[0]->gpu_diff(), (Dtype)1., weights_diff);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, K_, K_, -eta_,
        weights, temp_2_.gpu_data(), (Dtype)1., weights_diff);

    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, eta_,
        biased_input_.gpu_data(), top[0]->gpu_diff(), (Dtype)1.,
        weights_diff);
  }
  
  if (bias_term_ && this->param_propagate_down_[1]) { // Bias gradient
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, M_, (Dtype)1.,
        batch_multiplier_.gpu_data(), top[0]->gpu_diff(), (Dtype)0.,
        sum_top_diff_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, N_, K_, -eta_,
        sum_top_diff_.gpu_data(), weights, (Dtype)1., bias_diff);
  }

  if (propagate_down[0]) { // Data graident
    if (bottom.size() == 1) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)1.,
          weights, competition_matrix_.gpu_data(), (Dtype)0.,
          temp_3_.mutable_gpu_diff());
      caffe_gpu_axpby(temp_3_.count(), eta_, weights, -eta_,
          temp_3_.mutable_gpu_diff());
      caffe_gpu_axpy(temp_3_.count(), (Dtype)1., weights,
          temp_3_.mutable_gpu_diff());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
          top[0]->gpu_diff(), temp_3_.gpu_diff(), (Dtype)0.,
          bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, eta_,
          top[0]->gpu_diff(), weights, (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }

  if (propagate_down[1] && bottom.size() == 2) { // Activity gradient
    caffe_copy(bottom[1]->count(), top[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, -eta_,
        top[0]->gpu_diff(), competition_matrix_.gpu_data(), (Dtype)1., 
        bottom[1]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseUnitLayer);

} // namespace caffe
