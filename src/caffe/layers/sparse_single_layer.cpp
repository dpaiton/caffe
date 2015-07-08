#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseSingleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  eta_            = this->layer_param_.sparse_single_param().eta();
  lambda_         = this->layer_param_.sparse_single_param().lambda();

  bias_term_ = this->layer_param_.sparse_single_param().bias_term();

  //Find variables B_, L_, M_
  B_ = bottom[0]->shape(0);
  L_ = bottom[0]->count(1); // pixels per batch (all dim after batch are flattened)
  M_ = bottom[1]->count(1);

  // Allocate weights
  if (bias_term_) {
      this->blobs_.resize(2); // phi & bias
  } else {
      this->blobs_.resize(1); // only one stored param, phi
  }

  // Gradient switch for each param
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Intialize weights
  vector<int> weight_shape(2);
  weight_shape[0] = M_;
  weight_shape[1] = L_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  // Fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.sparse_single_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // Initialize & fill bias term
  if (bias_term_) {
      vector<int> bias_shape(1, L_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.sparse_single_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
  } 
}

template <typename Dtype>
void SparseSingleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Set size of biased_input_ (BxL)
  biased_input_.Reshape(bottom[0]->shape());

  // Set size of competition matrix (G is of dim MxM)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = M_;
  competition_matrix_shape[1] = M_;
  competition_matrix_.Reshape(competition_matrix_shape);

  backprop_multiplier_.Reshape(competition_matrix_shape);
  caffe_set(backprop_multiplier_.count(), (Dtype)1.,
            backprop_multiplier_.mutable_cpu_data());
  
  // Set size of top blob (BxM)
  vector<int> top_shape(2);
  top_shape[0] = B_;
  top_shape[1] = M_;
  top[0]->Reshape(top_shape);

  if (bias_term_) {
    vector<int> batch_mult_shape(1, B_);
    batch_multiplier_.Reshape(batch_mult_shape);
    caffe_set(B_, (Dtype)1., batch_multiplier_.mutable_cpu_data());
  }

  // Forward pass variable
  excitatory_input_.Reshape(top[0]->shape());

  // Backward pass variables
  vector<int> temp_1_shape(2);
  temp_1_shape[0] = B_;
  temp_1_shape[1] = L_;
  temp_1_.Reshape(temp_1_shape);

  vector<int> temp_2_shape(2);
  temp_2_shape[0] = B_;
  temp_2_shape[1] = M_;
  temp_2_.Reshape(temp_2_shape);

  vector<int> sum_top_diff_shape(1,M_);
  sum_top_diff_.Reshape(sum_top_diff_shape);
}

template <typename Dtype>
void SparseSingleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* weights = this->blobs_[0]->cpu_data(); // phi

  // Replicate bias vector into batch x pixel matrix, store in temp_1_
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, L_, 1, (Dtype)1.,
                batch_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)0.,
                temp_1_.mutable_cpu_data());

  // Subtract bias values from input
  caffe_sub(bottom[0]->count(), bottom[0]->cpu_data(), temp_1_.cpu_data(), 
            biased_input_.mutable_cpu_data());

  // Inhibition matrix (G matrix)
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, M_, L_,
          (Dtype)1., weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  // ext - excitatory input
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, M_, L_, eta_,
          biased_input_.cpu_data(), weights, (Dtype)0.,
          excitatory_input_.mutable_cpu_data());
  
  // a = bottom[1] + eta_ ( (x - b) phi - bottom[1] phi phi^T - lambda_ sgn(bottom[1]) )
  // top[0] = bottom[1] + eta_ ( excitatory_input_ - bottom[1] competition_matrix_ - lambda_ sgn(bottom[1]) )

  // Compute bottom[1] competition_matrix_ and store in temp_2_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_,
          -eta_, bottom[1]->cpu_data(), competition_matrix_.cpu_data(),
          (Dtype)0., temp_2_.mutable_cpu_data());

  // Add excitatory input to previous activity, store in top
  caffe_add(top[0]->count(), excitatory_input_.cpu_data(), bottom[1]->cpu_data(), top[0]->mutable_cpu_data());

  // Add -eta inhibition to top
  caffe_add(top[0]->count(), top[0]->cpu_data(), temp_2_.cpu_data(), top[0]->mutable_cpu_data());

  // Compute sgn on previous a's, store in temp_2_
  caffe_cpu_sign(top[0]->count(), bottom[1]->cpu_data(), temp_2_.mutable_cpu_data());

  // Sub lambda term from top
  caffe_axpy(top[0]->count(), -lambda_, temp_2_.cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SparseSingleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    //weight gradient should be:
    //     eta_ ( (s-b) - 2 a[t-1] phi)

    // compute (2 a[t-1] phi), store in temp_1_
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)2.,
                      bottom[1]->cpu_data(), this->blobs_[0]->cpu_data(),
                      (Dtype)0., temp_1_.mutable_cpu_data());

    // compute eta_ (s - b) - eta_ (2 a[t-1] phi) , store in temp_1_
    caffe_cpu_axpby(temp_1_.count(), eta_, biased_input_.cpu_data(),
                  -eta_, temp_1_.mutable_cpu_data());

    // compute top_diff^T temp_1_, store in weight_diff
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, L_, B_, (Dtype)1.,
                       top[0]->cpu_diff(), temp_1_.cpu_data(), (Dtype)1.,
                       this->blobs_[0]->mutable_cpu_diff());

/*
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, L_, B_, (Dtype)1.,
                        top[0]->cpu_diff(), biased_input_.cpu_data(), (Dtype)0.,
                        this->blobs_[0]->mutable_cpu_diff());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)-2.,
                      bottom[1]->cpu_data(), this->blobs_[0]->cpu_data(),
                      (Dtype)0., temp_1_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)1.,
                       top[0]->cpu_diff(), temp_1_.cpu_data(),
                       (Dtype)1., this->blobs_[0]->mutable_cpu_diff()); 
*/

    // Bias
    // sum top over B, then multiply by phi
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, M_, B_, (Dtype)1.,
                        batch_multiplier_.cpu_data(), top[0]->cpu_diff(), 
                        (Dtype)0., sum_top_diff_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, L_, M_, (Dtype)-eta_,
                        sum_top_diff_.cpu_data(), this->blobs_[0]->cpu_data(),
                        (Dtype)1., this->blobs_[1]->mutable_cpu_diff());

    // Input_0
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)eta_,
        top[0]->cpu_diff(), this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());

    // Input_1
    // diff = tdiff ( 1 - eta_ G )
    caffe_axpy(backprop_multiplier_.count(), -eta_, competition_matrix_.cpu_data(),
            backprop_multiplier_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_, 
                    (Dtype)1., top[0]->cpu_diff(), backprop_multiplier_.cpu_data(),
                    (Dtype)0., bottom[1]->mutable_cpu_diff());

}

#ifdef CPU_ONLY
STUB_GPU(SparseSingleLayer);
#endif

INSTANTIATE_CLASS(SparseSingleLayer);
REGISTER_LAYER_CLASS(SparseSingle);

}  // namespace caffe
