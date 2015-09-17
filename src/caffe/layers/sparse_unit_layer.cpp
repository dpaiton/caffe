#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseUnitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  eta_       = this->layer_param_.sparse_unit_param().eta();
  lambda_    = this->layer_param_.sparse_unit_param().lambda();
  bias_term_ = this->layer_param_.sparse_unit_param().bias_term();

  M_ = bottom[0]->shape(0); // batch size
  N_ = bottom[0]->count(1); // num pixels per batch
  if (bottom.size() == 2) {
      K_ = bottom[1]->count(1); // num elements (features)
  } else {
      K_ = this->layer_param_.sparse_unit_param().num_elements();
  }

  if (bias_term_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }

  //Gradient switch for each param
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  vector<int> weight_shape(2);
  weight_shape[0] = N_;
  weight_shape[1] = K_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.sparse_unit_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  if (bias_term_) { 
    vector<int> bias_shape(1,N_);
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.sparse_unit_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
}

template <typename Dtype>
void SparseUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(2);
  top_shape[0] = M_;
  top_shape[1] = K_;
  top[0]->Reshape(top_shape);

  biased_input_.Reshape(bottom[0]->shape());

  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = K_;
  competition_matrix_shape[1] = K_;
  competition_matrix_.Reshape(competition_matrix_shape);

  if (bias_term_) {
    vector<int> batch_mult_shape(1, M_);
    batch_multiplier_.Reshape(batch_mult_shape);
    caffe_set(M_, (Dtype)1., batch_multiplier_.mutable_cpu_data());
  }

  if (bottom.size() == 1) { // no previous activity specified
    previous_activity_.Reshape(top_shape); 
    caffe_set(previous_activity_.count(), (Dtype)0., previous_activity_.mutable_cpu_data());
  }

  vector<int>temp_shape(2);
  temp_shape[0] = M_;
  temp_shape[1] = N_;
  temp_1_.Reshape(temp_shape);
  temp_shape[0] = K_;
  temp_shape[1] = K_;
  temp_2_.Reshape(temp_shape);
  if (bottom.size() == 1) {
    temp_shape[0] = N_;
    temp_shape[1] = K_;
    temp_3_.Reshape(temp_shape);
  }

  vector<int> sum_top_diff_shape(1, K_);
  sum_top_diff_.Reshape(sum_top_diff_shape);
}

template <typename Dtype>
void SparseUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* weights    = this->blobs_[0]->cpu_data(); // phi    :: N_xK_
  const Dtype* a_past;                                   // a(t-1) :: M_xK_
  Dtype* mutable_top_data = top[0]->mutable_cpu_data();  // output :: M_xK_

  if (bottom.size() == 1) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        bottom[0]->cpu_data(), weights, (Dtype)0., previous_activity_.mutable_cpu_data());
    a_past = previous_activity_.cpu_data();
  } else {
    a_past = bottom[1]->cpu_data();
  }

  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data(); // bias   :: 1xN_
    // Replicate bias vector into M_xN_ matrix, store in temp_1_
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        batch_multiplier_.cpu_data(), bias, (Dtype)0., temp_1_.mutable_cpu_data());
    // Subtract bias values from input
    caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), temp_1_.cpu_data(),
        biased_input_.mutable_cpu_data());
  } else {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), biased_input_.mutable_cpu_data());
  }

  // competition_matrix_ = phi^T phi
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, N_, (Dtype)1.,
      weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  // top = sgn(a[t-1])
  caffe_cpu_sign(top[0]->count(), a_past, mutable_top_data);

  // top = a[t-1] G + lambda_ sgn(a[t-1])
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
      a_past, competition_matrix_.cpu_data(), lambda_, mutable_top_data);

  // top = (s-b) w - (a[t-1] G + lambda_ sgn(a[t-1]))
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      biased_input_.cpu_data(), weights, (Dtype)-1., mutable_top_data);

  // a[t-1] + eta_ ((s-b) w - a[t-1] G - lambda_ sgn(a[t-1]))
  caffe_cpu_axpby(top[0]->count(), (Dtype)1., a_past, eta_, mutable_top_data);
}

template <typename Dtype>
void SparseUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* weights = this->blobs_[0]->cpu_data();
  
  if (this->param_propagate_down_[0]) { // Weight gradient
    const Dtype* a_past;
    if (bottom.size() == 1) {
      a_past = previous_activity_.cpu_data();
    } else {
      a_past = bottom[1]->cpu_data();
    }
    Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        a_past, weights, (Dtype)0., temp_1_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
        a_past, top[0]->cpu_diff(), (Dtype)0., temp_2_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, -eta_,
        temp_1_.cpu_data(), top[0]->cpu_diff(), (Dtype)1., weights_diff);

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, K_, K_, -eta_,
        weights, temp_2_.cpu_data(), (Dtype)1., weights_diff);

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, eta_,
        biased_input_.cpu_data(), top[0]->cpu_diff(), (Dtype)1.,
        weights_diff);
  }
  
  if (bias_term_ && this->param_propagate_down_[1]) { // Bias gradient
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, M_, (Dtype)1.,
        batch_multiplier_.cpu_data(), top[0]->cpu_diff(), (Dtype)0.,
        sum_top_diff_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, N_, K_, -eta_,
        sum_top_diff_.cpu_data(), weights, (Dtype)1., bias_diff);
  }

  if (propagate_down[0]) { // Data graident
    if (bottom.size() == 1) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)1.,
          weights, competition_matrix_.cpu_data(), (Dtype)0.,
          temp_3_.mutable_cpu_diff());
      caffe_cpu_axpby(temp_3_.count(), eta_, weights, -eta_,
          temp_3_.mutable_cpu_diff());
      caffe_axpy(temp_3_.count(), (Dtype)1., weights,
          temp_3_.mutable_cpu_diff());
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
          top[0]->cpu_diff(), temp_3_.cpu_diff(), (Dtype)0.,
          bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, eta_,
          top[0]->cpu_diff(), weights, (Dtype)0., bottom[0]->mutable_cpu_diff());
    } 
  }

  if (propagate_down[1] && bottom.size() == 2) { // Activity gradient
    caffe_copy(bottom[1]->count(), top[0]->cpu_diff(), bottom[1]->mutable_cpu_diff());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, -eta_,
        top[0]->cpu_diff(), competition_matrix_.cpu_data(), (Dtype)1., 
        bottom[1]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(SparseUnitLayer);
#endif

INSTANTIATE_CLASS(SparseUnitLayer);
REGISTER_LAYER_CLASS(SparseUnit);

} // namespace caffe
