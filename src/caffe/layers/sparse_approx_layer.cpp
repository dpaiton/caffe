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

  //Find variables M_, K_, N_
  M_ = bottom[0]->shape(0); // batch
  N_ = bottom[0]->count(1); // pixels per batch (all dim after batch are flattened)
  K_ = this->layer_param_.sparse_approx_param().num_elements(); // num elements

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
  weight_shape[0] = N_;
  weight_shape[1] = K_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  // Fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.sparse_approx_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  // Initialize & fill bias term
  if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
  } 
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Set size of biased_input_ (M_xN_)
  biased_input_.Reshape(bottom[0]->shape());

  // Set size of activity_history_ (TxM_xK_), where T = num_iterations_
  vector<int> activity_history_shape(3);
  activity_history_shape[0] = num_iterations_;
  activity_history_shape[1] = M_;
  activity_history_shape[2] = K_;
  activity_history_.Reshape(activity_history_shape);
  caffe_set(activity_history_.count(), (Dtype)0., activity_history_.mutable_cpu_data());

  // Set size of competition matrix (G is of dim K_xK_)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = K_;
  competition_matrix_shape[1] = K_;
  competition_matrix_.Reshape(competition_matrix_shape);

  // Set size of top blob (M_xK_)
  vector<int> top_shape(2);
  top_shape[0] = M_;
  top_shape[1] = K_;
  top[0]->Reshape(top_shape);

  // Multiplier for summing (or replicating) over batch
  if (bias_term_) {
    vector<int> batch_mult_shape(1, M_);
    batch_multiplier_.Reshape(batch_mult_shape);
    caffe_set(M_, (Dtype)1., batch_multiplier_.mutable_cpu_data());
  }

  // Create identity matrix
  identity_matrix_.Reshape(competition_matrix_shape);
  caffe_set(identity_matrix_.count(), (Dtype)0.,
    identity_matrix_.mutable_cpu_data());
  for (int i = 0; i < K_; ++i) {
    identity_matrix_.mutable_cpu_data()[i*K_ + i] = 1;
  }

  excitatory_input_.Reshape(top[0]->shape());

  vector<int> temp_shape(2);
  temp_shape[0] = M_;
  temp_shape[1] = N_;
  temp_1_.Reshape(temp_shape);
  temp_shape[0] = K_;
  temp_shape[1] = K_;
  temp_2_.Reshape(temp_shape);

  backprop_multiplier_.Reshape(competition_matrix_shape);

  vector<int> sum_top_diff_shape(1,K_);
  sum_top_diff_.Reshape(sum_top_diff_shape);

  temp_tdiff_.Reshape(top[0]->shape());
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* input    = bottom[0]->cpu_data();       // input  (M_xN_)
  const Dtype* top_data = top[0]->cpu_data();          // output (M_xK_)
  const Dtype* weights  = this->blobs_[0]->cpu_data(); // phi    (N_xK_)
  const Dtype* bias     = this->blobs_[1]->cpu_data(); // bias   (1xN_)

  //Replicate bias vector into M_ x N_ matrix, store in temp_1_
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
    batch_multiplier_.cpu_data(), bias, (Dtype)0., temp_1_.mutable_cpu_data());

  // Subtract bias values from input
  caffe_sub(bottom[0]->count(), input, temp_1_.cpu_data(),
    biased_input_.mutable_cpu_data()); 

  // Competition matrix (G matrix)
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, N_, (Dtype)1.,
    weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  // Excitatory input
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
    biased_input_.cpu_data(), weights, (Dtype)0., excitatory_input_.mutable_cpu_data());

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

      // TODO: Finish thresholding code
      //// Threshold previous activities
      //// Currently implements rectified soft threshold.
      //for (int i = 0; i < M_*N_; ++i) {
      //    if (const_a_past[i] < gamma_) {
      //        caffe_set(1, (Dtype)0., mutable_a_past + i);
      //    }
      //}

      // Add ext value to current hist slot
      caffe_add(top[0]->count(), excitatory_input_.cpu_data(), const_a_current, mutable_a_current);

      // Compute ext - a[iteration-1] G
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_,
              (Dtype)-1., const_a_past, competition_matrix_.cpu_data(), (Dtype)1.,
              mutable_a_current);

     // Compute sign on previous a's, store in top temporarily
     caffe_cpu_sign(top[0]->count(), const_a_past, top[0]->mutable_cpu_data());

     //Compute -lambda_ sign(a) + (ext - a[iteration-1] G)
     caffe_cpu_axpby(top[0]->count(), -lambda_, top_data, (Dtype)1., mutable_a_current);

     // Add previous activities to eta_ [...]
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

    // GradientStats
    stringstream ss;
    ss << "Iter\telem_mean\t\tgrad_mean\t\tgrad_var\n";

    const Dtype* weights = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff   = bottom[0]->mutable_cpu_diff();
    Dtype* weights_diff  = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bias_diff     = this->blobs_[1]->mutable_cpu_diff();

    // Clear bottom diff
    caffe_set(bottom[0]->count(), (Dtype)0., bottom_diff);

    // Compute I - eta_ G, put in backprop_multiplier
    caffe_copy(backprop_multiplier_.count(), identity_matrix_.cpu_data(),
      backprop_multiplier_.mutable_cpu_data());
    caffe_axpy(backprop_multiplier_.count(), -eta_, competition_matrix_.cpu_data(),
      backprop_multiplier_.mutable_cpu_data());
    
    // First iteration holds top_diff
    caffe_copy(top[0]->count(), top[0]->cpu_diff(), temp_tdiff_.mutable_cpu_diff());

    // GradientStats
    Dtype data_mean = 0;
    Dtype grad_mean = 0;
    //Dtype grad_std  = 0;
    Dtype temp      = 0;
    for (int iteration = num_iterations_-1; iteration >= 0; --iteration) {

        // GradientStats
        ss << iteration << "\t";

        // Weight gradient
        if (this->param_propagate_down_[0]) {
            if (iteration != 0) {
                const Dtype* const_a_past;
                const_a_past = activity_history_.cpu_data() +
                               activity_history_.offset(iteration-1);

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                  const_a_past, weights, (Dtype)0., temp_1_.mutable_cpu_data());

                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
                  const_a_past, temp_tdiff_.cpu_diff(), (Dtype)0., temp_2_.mutable_cpu_data());

                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, -eta_,
                  temp_1_.cpu_data(), temp_tdiff_.cpu_diff(), (Dtype)1., weights_diff);

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, K_, K_, -eta_,
                  weights, temp_2_.cpu_data(), (Dtype)1., weights_diff);

            }
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, eta_,
              biased_input_.cpu_data(), temp_tdiff_.cpu_diff(), (Dtype)1.,
              weights_diff);
            
            // GradientStats
            for (int el = 0; el < N_; ++el) {
                data_mean += (weights[el] > Dtype(0.)) ? weights[el] : -weights[el];
                grad_mean += (weights_diff[el] > Dtype(0.)) ? weights_diff[el] : -weights_diff[el];
            }
            data_mean /= this->blobs_[0]->count();
            grad_mean /= this->blobs_[0]->count();
            Dtype grad_var = 0;
            for (int i = 0; i < this->blobs_[0]->count(); ++i) {
                temp = (weights_diff[i] > Dtype(0.)) ? weights_diff[i] : -weights_diff[i];
                grad_var += (temp - grad_mean) * (temp - grad_mean);
            }
            grad_var /= this->blobs_[0]->count();
            //grad_std = sqrt(grad_var);
            ss << std::scientific << data_mean << "\t\t";
            ss << std::scientific << grad_mean << "\t\t";
            ss << std::scientific << grad_var << "\t\t";
        }

        // Bias gradient
        if (bias_term_ && this->param_propagate_down_[1]) {
            // sum top over B
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, M_, (Dtype)1.,
              batch_multiplier_.cpu_data(), temp_tdiff_.cpu_diff(), (Dtype)0.,
              sum_top_diff_.mutable_cpu_data());

            // multiply by phi
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, N_, K_, -eta_,
              sum_top_diff_.cpu_data(), weights, (Dtype)1., bias_diff);
        }
    
        // Input gradient
        if (propagate_down[0]) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, eta_,
              temp_tdiff_.cpu_diff(), weights, (Dtype)1., bottom_diff);
        }

        // Update tdiff
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
          top[0]->cpu_diff(), backprop_multiplier_.cpu_data(), (Dtype)0.,
          temp_tdiff_.mutable_cpu_diff());
        
        caffe_copy(temp_tdiff_.count(), temp_tdiff_.cpu_diff(), top[0]->mutable_cpu_diff());
        
	ss << "\n";
        stats_string_ = ss.str();
    }
}

#ifdef CPU_ONLY
STUB_GPU(SparseApproxLayer);
#endif

INSTANTIATE_CLASS(SparseApproxLayer);
REGISTER_LAYER_CLASS(SparseApprox);

}  // namespace caffe
