#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseApproxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  num_iterations_ = this->layer_param_.sparse_approx_param().num_iterations();
  eta_            = this->layer_param_.sparse_approx_param().eta();
  lambda_         = this->layer_param_.sparse_approx_param().lambda();
  gamma_          = this->layer_param_.sparse_approx_param().gamma();

  bias_term_ = this->layer_param_.sparse_approx_param().bias_term();

  //Find variables L_, M_
  L_ = bottom[0]->count(1); // pixels per batch (all dim after batch are flattened)
  M_ = this->layer_param_.sparse_approx_param().num_elements();

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
      this->layer_param_.sparse_approx_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // Initialize & fill bias term
  if (bias_term_) {
      vector<int> bias_shape(1, L_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.sparse_approx_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
  } 
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  B_ = bottom[0]->shape(0);

  // Set size of biased_input_ (BxL)
  biased_input_.Reshape(bottom[0]->shape());

  // Set size of activity history matrix (Tx(B_*M_), where T = num_iterations
  vector<int> activity_history_shape(3);
  activity_history_shape[0] = num_iterations_;
  activity_history_shape[1] = B_;
  activity_history_shape[2] = M_;
  activity_history_.Reshape(activity_history_shape);
  caffe_set(activity_history_.count(), (Dtype)0., activity_history_.mutable_cpu_data());

  // Set size of competition matrix (G is of dim MxM)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = M_;
  competition_matrix_shape[1] = M_;
  competition_matrix_.Reshape(competition_matrix_shape);

  backprop_multiplier_.Reshape(competition_matrix_shape);
  
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
  vector<int> temp_shape(2);
  temp_shape[0] = B_;
  temp_shape[1] = L_;
  temp_1_.Reshape(temp_shape);

  vector<int> sum_top_diff_shape(1,M_);
  sum_top_diff_.Reshape(sum_top_diff_shape);

  temp_tdiff_.Reshape(top[0]->shape());
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, M_, L_, (Dtype)1.,
          biased_input_.cpu_data(), weights, (Dtype)0.,
          excitatory_input_.mutable_cpu_data());

  // First iteration
  caffe_copy(top[0]->count(), excitatory_input_.cpu_data(),
            activity_history_.mutable_cpu_data());
  
  caffe_scal(top[0]->count(), eta_, activity_history_.mutable_cpu_data());

  // Next iterations
  for (int iteration = 1; iteration < num_iterations_; ++iteration) {
    // Set up pointers
    Dtype* mutable_a_current     = activity_history_.mutable_cpu_data() + activity_history_.offset(iteration);
    //Dtype* mutable_a_past        = activity_history_.mutable_cpu_data() + activity_history_.offset(iteration-1);
    const Dtype* const_a_current = activity_history_.cpu_data() + activity_history_.offset(iteration);
    const Dtype* const_a_past    = activity_history_.cpu_data() + activity_history_.offset(iteration-1);

    // Add ext to current history slot
    caffe_add(top[0]->count(), excitatory_input_.cpu_data(), const_a_current, mutable_a_current);

    // Compute ext - a[iteration-1] G
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_,
            (Dtype)-1., const_a_past, competition_matrix_.cpu_data(), (Dtype)1.,
            mutable_a_current);

    // Compute sign on previous a's, store in top temporarily
    caffe_cpu_sign(top[0]->count(), const_a_past, top[0]->mutable_cpu_data());

    // Compute -lambda_ sign(a) + (ext - a[iteration-1] G)
    caffe_cpu_axpby(top[0]->count(), -lambda_, top[0]->cpu_data(), 
                (Dtype)1., mutable_a_current);

    // Add previous activities to eta_ (current activities)
    caffe_cpu_axpby(top[0]->count(), (Dtype)1., const_a_past, eta_, mutable_a_current);
  }

  // Store latest activity history into top for output
  caffe_copy(top[0]->count(),
          activity_history_.cpu_data() + activity_history_.offset(num_iterations_-1),
          top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    // set backprop multiplier to identity matrix
    caffe_set(backprop_multiplier_.count(), (Dtype)0.,
              backprop_multiplier_.mutable_cpu_data());

    for (int i = 0; i < M_; ++i) {
      backprop_multiplier_.mutable_cpu_data()[i*M_+i] = 1;
    }

    // Scalar for top_diff through time is -eta_ G
    // Compute I - eta_ G
    caffe_axpy(backprop_multiplier_.count(), -eta_, competition_matrix_.cpu_data(),
            backprop_multiplier_.mutable_cpu_data());

    caffe_copy(top[0]->count(), top[0]->cpu_diff(), temp_tdiff_.mutable_cpu_diff());

    for (int iteration = num_iterations_-1; iteration >= 0; --iteration) {
        // Weights
        // weight gradient should be:
        //     eta_ ( (s-b) - 2 a[t-1] phi)
        const Dtype* const_a_past;
        if (iteration == 0) {
            const_a_past = activity_history_.cpu_data();
        } else {
            const_a_past = activity_history_.cpu_data() +
                                        activity_history_.offset(iteration-1);
        }

        // compute (2 a[t-1] phi), store in temp_1_
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)2.,
                          const_a_past, this->blobs_[0]->cpu_data(),
                          (Dtype)0., temp_1_.mutable_cpu_data());

        // compute eta_ (s - b) - eta_ (2 a[t-1] phi) , store in temp_1_
        caffe_cpu_axpby(temp_1_.count(), eta_, biased_input_.cpu_data(),
                      -eta_, temp_1_.mutable_cpu_data());

        // compute top_diff^T [...], store in weight_diff
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, L_, B_, (Dtype)1.,
                           temp_tdiff_.cpu_diff(), temp_1_.cpu_data(), (Dtype)1.,
                           this->blobs_[0]->mutable_cpu_diff());

        // Bias
        // sum top over B
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, M_, B_, (Dtype)1.,
                            batch_multiplier_.cpu_data(), temp_tdiff_.cpu_diff(), 
                            (Dtype)0., sum_top_diff_.mutable_cpu_data());
        // multiply by phi
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, L_, M_, (Dtype)-eta_,
                            sum_top_diff_.cpu_data(), this->blobs_[0]->cpu_data(),
                            (Dtype)1., this->blobs_[1]->mutable_cpu_diff());

        // Input
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, L_, M_, (Dtype)eta_,
            temp_tdiff_.cpu_diff(), this->blobs_[0]->cpu_data(), (Dtype)1.,
            bottom[0]->mutable_cpu_diff());

        // Update diff
        // tdiff = tdiff (I - eta_ G)
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_, 
                        (Dtype)1., top[0]->cpu_diff(), backprop_multiplier_.cpu_data(),
                        (Dtype)0., temp_tdiff_.mutable_cpu_diff());
        caffe_copy(temp_tdiff_.count(), temp_tdiff_.cpu_diff(), top[0]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(SparseApproxLayer);
#endif

INSTANTIATE_CLASS(SparseApproxLayer);
REGISTER_LAYER_CLASS(SparseApprox);

}  // namespace caffe
