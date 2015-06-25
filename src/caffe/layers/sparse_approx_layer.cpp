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

  //Find variables B_, L_, M_
  B_ = bottom[0]->shape(0);
  L_ = bottom[0]->count() / bottom[0]->shape(0); // pixels per batch
  M_ = this->layer_param_.sparse_approx_param().num_elements();

  // Allocate weights
  if (bias_term_) {
      this->blobs_.resize(2); // phi & bias
  } else {
      this->blobs_.resize(1); // only one stored param, phi
  }

  // Intialize weights
  vector<int> weight_shape(2);
  weight_shape[0] = L_;
  weight_shape[1] = M_;
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
        this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
  } 
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Set size of biased_input_ (BxL)
  biased_input_.Reshape(bottom[0]->shape());

  // Set size of activity history matrix (TxM, where T = num_iterations
  vector<int> activity_history_shape(2);
  activity_history_shape[0] = num_iterations_;
  activity_history_shape[1] = B_*M_;
  activity_history_.Reshape(activity_history_shape);
  caffe_set(activity_history_.count(), (Dtype)0., activity_history_.mutable_cpu_data());

  // Set size of competition matrix (G is of dim MxM)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = M_;
  competition_matrix_shape[1] = M_;
  competition_matrix_.Reshape(competition_matrix_shape);

  // Set size of top blob (BxM)
  vector<int> top_shape(2);
  top_shape[0] = B_;
  top_shape[1] = M_;
  top[0]->Reshape(top_shape);
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());

  if (bias_term_) {
    vector<int> batch_mult_shape(1, B_);
    batch_multiplier_.Reshape(batch_mult_shape);
    caffe_set(B_, (Dtype)1., batch_multiplier_.mutable_cpu_data());
  }

  // Backward pass variables
  vector<int> temp_shape(2);
  temp_shape[0] = B_;
  temp_shape[1] = L_;
  temp_1_.Reshape(temp_shape);
  temp_2_.Reshape(temp_shape);

  vector<int> sum_top_diff_shape(1,M_);
  sum_top_diff_.Reshape(sum_top_diff_shape);
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // f(a) = a + eta_ [ (x-b) phi - a phi**T phi - lambda_ sgn(a)]
  //                   |________|    |________|   |____________|
  //                      ext            g              c
  //
  // ext  = gemm(x,phi)       (BxL) * (LxM) = (BxM)  ->  temp_0
  // g    = gemm(phi**T,phi)  (MxL) * (LxM) = (MxM)  ->  competition_matrix_
  // ag   = gemm(a,g)        (BxM) * (MxM) = (BxM)  ->  top
  // e_ga = sub(ext, ag)
  // f(a) = add(a, eta_ * add(e_ga, lambda_ sgn(a)))
  // 
  // weights     //  output    //  input     //  bias
  // phi -- LxM  //  a -- BxM  //  x -- BxL  //  b -- 1xL

  // Subtract bias values from input
  for (int batch=0; batch < bottom[0]->shape(0); batch++) { // same bias is applied to each batch item
      caffe_sub(L_, bottom[0]->cpu_data() + bottom[0]->offset(batch),
              this->blobs_[1]->cpu_data(),
              biased_input_.mutable_cpu_data() + biased_input_.offset(batch)); // x = x - bias
  }

  const Dtype* weights = this->blobs_[0]->cpu_data(); // phi

  //temp_0 holds ext values
  Blob<Dtype> temp_0;
  temp_0.Reshape(top[0]->shape());

  // First iteration
  // g - inhibition matrix
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, M_, L_,
          (Dtype)1., weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  // ext - excitatory input
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, L_,
          (Dtype)1., biased_input_.cpu_data(), weights, (Dtype)0.,
          temp_0.mutable_cpu_data());

  caffe_copy(top[0]->count(), temp_0.cpu_data(), activity_history_.mutable_cpu_data());
  
  // f(a) - top output
  caffe_scal(top[0]->count(), eta_, activity_history_.mutable_cpu_data());

  // Next iterations
  for (int iteration = 1; iteration < num_iterations_; iteration++) {
      // Set up pointers
      Dtype* mutable_a_current     = activity_history_.mutable_cpu_data() + activity_history_.offset(iteration);
      //Dtype* mutable_a_past        = activity_history_.mutable_cpu_data() + activity_history_.offset(iteration-1);
      const Dtype* const_a_current = activity_history_.cpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_past    = activity_history_.cpu_data() + activity_history_.offset(iteration-1);

      // TODO: Finish thresholding code
      //// Threshold previous activities
      //// Currently implements rectified soft threshold.
      //for (int i = 0; i < M_*B_; i++) {
      //    if (const_a_past[i] < gamma_) {
      //        caffe_set(1, (Dtype)0., mutable_a_past + i);
      //    }
      //}

      // Add ext value to output, store in current history slot
      caffe_add(top[0]->count(), temp_0.cpu_data(), const_a_current, mutable_a_current);

      // Compute ext - a[iteration-1] g, store in current history slot
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_,
              (Dtype)-1., const_a_past, competition_matrix_.cpu_data(), (Dtype)1.,
              mutable_a_current);

     // Compute sign on previous a's, store in top temporarily
     caffe_cpu_sign(top[0]->count(), const_a_past, top[0]->mutable_cpu_data());

     // lambda_ * sign(a), store in top temporarily
     caffe_scal(top[0]->count(), lambda_, top[0]->mutable_cpu_data()); 

     // Store [ ... ] in current history slot
     caffe_sub(top[0]->count(), const_a_current, top[0]->cpu_data(), 
             mutable_a_current);

     // eta_ * [ ... ], store in current hist slot
     caffe_scal(top[0]->count(), eta_, mutable_a_current);

     // Add previous activities to current acticities, store in current slot
     caffe_add(top[0]->count(), const_a_past, const_a_current, mutable_a_current);
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

    // weights     //  output    //  input     //  bias
    // phi -- LxM  //  a -- BxM  //  x -- BxL  //  b -- 1xL
    //
    // top_diff should be (BxM):
    //     {-1/P (s - [a + eta ((s - b) phi - a phi^T phi - 
    //     lambda sgn(a))]} phi

    // Weight gradient
    if (this->param_propagate_down_[0]) {
        // weight gradient should be:
        //      eta [ (s-b) - 2 a phi^T]

        // compute [a phi^T], store in temp_1_
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, L_, M_, (Dtype)1.,
                          top[0]->cpu_data(), this->blobs_[0]->cpu_data(),
                          (Dtype)0., temp_1_.mutable_cpu_data());

        caffe_scal(temp_1_.count(), (Dtype)2., temp_1_.mutable_cpu_data());

        // compute (s - b) - [2a phi^T], store in temp_2_
        caffe_sub(biased_input_.count(), biased_input_.cpu_data(), 
                temp_1_.cpu_data(), temp_2_.mutable_cpu_data());

        caffe_scal(temp_2_.count(), eta_, temp_2_.mutable_cpu_data());

        // compute [...]^T top_diff, store in weight_diff
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L_, M_, B_, (Dtype)1.,
                           temp_2_.cpu_data(), top[0]->cpu_diff(), (Dtype)0.,
                           this->blobs_[0]->mutable_cpu_diff());
    }

    // Bias gradient
    if (bias_term_ && this->param_propagate_down_[1]) {
        // Sum top_diff over B axis
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, M_, B_, (Dtype)1.,
                            batch_multiplier_.cpu_data(), top[0]->cpu_diff(), 
                            (Dtype)0., sum_top_diff_.mutable_cpu_data());

        caffe_scal(sum_top_diff_.count(), -eta_, sum_top_diff_.mutable_cpu_data());

        // -eta * sum(top_diff) * phi^T 
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, L_, M_, (Dtype)1.,
                            sum_top_diff_.cpu_data(), this->blobs_[0]->cpu_data(),
                            (Dtype)0., this->blobs_[1]->mutable_cpu_diff());
    }
    
    // Input gradient
    if (propagate_down[0]) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, B_, L_, M_, (Dtype)1.,
                            top[0]->cpu_diff(), this->blobs_[0]->cpu_data(), (Dtype)0.,
                            bottom[0]->mutable_cpu_diff());

        caffe_scal(bottom[0]->count(), eta_, bottom[0]->mutable_cpu_data());
    }
}

#ifdef CPU_ONLY
STUB_GPU(SparseApproxLayer);
#endif

INSTANTIATE_CLASS(SparseApproxLayer);
REGISTER_LAYER_CLASS(SparseApprox);

}  // namespace caffe
