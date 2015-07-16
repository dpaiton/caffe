#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* input    = bottom[0]->gpu_data();       // input
  const Dtype* top_data = top[0]->gpu_data();          // output
  const Dtype* weights  = this->blobs_[0]->gpu_data(); // phi
  const Dtype* bias     = this->blobs_[1]->gpu_data(); // bias

  //Replicate bias vector into M_ x K_ matrix, store in temp_1_
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
    batch_multiplier_.gpu_data(), bias, (Dtype)0., temp_1_.mutable_gpu_data());

  // Subtract bias values from input
  caffe_gpu_sub(bottom[0]->count(), input, temp_1_.gpu_data(),
    biased_input_.mutable_gpu_data()); 

  // Competition matrix (G matrix)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_, (Dtype)1.,
    weights, weights, (Dtype)0., competition_matrix_.mutable_gpu_data());

  // Excitatory input
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
    biased_input_.gpu_data(), weights, (Dtype)0., excitatory_input_.mutable_gpu_data());

  // First iteration
  caffe_copy(top[0]->count(), excitatory_input_.gpu_data(),
    activity_history_.mutable_gpu_data());

  caffe_gpu_scal(top[0]->count(), eta_, activity_history_.mutable_gpu_data());

  // Next iterations
  for (int iteration = 1; iteration < num_iterations_; ++iteration) {
      // Set up pointers
      Dtype* mutable_a_current     = activity_history_.mutable_gpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_current = activity_history_.gpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_past    = activity_history_.gpu_data() + activity_history_.offset(iteration-1);

      // Add ext value to current hist slot
      caffe_gpu_add(top[0]->count(), excitatory_input_.gpu_data(), const_a_current, mutable_a_current);

      // Compute ext - a[iteration-1] G
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, N_,
              (Dtype)-1., const_a_past, competition_matrix_.gpu_data(), (Dtype)1.,
              mutable_a_current);

     // Compute sign on previous a's, store in top temporarily
     caffe_gpu_sign(top[0]->count(), const_a_past, top[0]->mutable_gpu_data());

     //Compute -lambda_ sign(a) + (ext - a[iteration-1] G)
     caffe_gpu_axpby(top[0]->count(), -lambda_, top_data, (Dtype)1., mutable_a_current);

     // Add previous activities to eta_ [...]
     caffe_gpu_axpby(top[0]->count(), (Dtype)1., const_a_past, eta_, mutable_a_current);
  }

  // Store latest activity history into top for output
  caffe_copy(top[0]->count(),
          activity_history_.gpu_data() + activity_history_.offset(num_iterations_-1),
          top[0]->mutable_gpu_data());
    
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* weights = this->blobs_[0]->gpu_data();
    Dtype* bottom_diff   = bottom[0]->mutable_gpu_diff();
    Dtype* weights_diff  = this->blobs_[0]->mutable_gpu_diff();
    Dtype* bias_diff     = this->blobs_[1]->mutable_gpu_diff();

    // Clear bottom diff
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom_diff);

    // Set backprop multiplier to identity matrix
    caffe_gpu_set(backprop_multiplier_.count(), (Dtype)0.,
      backprop_multiplier_.mutable_gpu_data());

    CUDA_KERNEL_LOOP(index, N_) {
        backprop_multiplier_.mutable_gpu_data()[index*N_ + index] = 1;
    }
    //for (int i = 0; i < N_; ++i) {
    //  backprop_multiplier_.mutable_gpu_data()[i*N_ + i] = 1;
    //}

    // Compute I - eta_ G, put in backprop_multiplier
    caffe_gpu_axpy(backprop_multiplier_.count(), -eta_, competition_matrix_.gpu_data(),
      backprop_multiplier_.mutable_gpu_data());
    
    // First iteration holds top_diff
    caffe_copy(top[0]->count(), top[0]->gpu_diff(), temp_tdiff_.mutable_gpu_diff());

    for (int iteration = num_iterations_-1; iteration >= 0; --iteration) {
        // Weight gradient
        if (this->param_propagate_down_[0]) {
            if (iteration != 0) {
                const Dtype* const_a_past;
                const_a_past = activity_history_.gpu_data() +
                               activity_history_.offset(iteration-1);

                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                  const_a_past, weights, (Dtype)0., temp_1_.mutable_gpu_data());

                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, N_, M_, (Dtype)1.,
                  temp_tdiff_.gpu_diff(), const_a_past, (Dtype)0., temp_2_.mutable_gpu_data());

                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, N_, -eta_,
                  temp_2_.gpu_data(), weights, (Dtype)1., weights_diff);

                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, -eta_,
                  temp_tdiff_.gpu_diff(), temp_1_.gpu_data(), (Dtype)1., weights_diff);

            }
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, eta_,
              temp_tdiff_.gpu_diff(), biased_input_.gpu_data(), (Dtype)1.,
              weights_diff);
        }

        // Bias gradient
        if (bias_term_ && this->param_propagate_down_[1]) {
            // sum top over B
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_, (Dtype)1.,
              batch_multiplier_.gpu_data(), temp_tdiff_.gpu_diff(), (Dtype)0.,
              sum_top_diff_.mutable_gpu_data());

            // multiply by phi
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, N_, -eta_,
              sum_top_diff_.gpu_data(), weights, (Dtype)1., bias_diff);
        }
    
        // Input gradient
        if (propagate_down[0]) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, eta_,
              temp_tdiff_.gpu_diff(), weights, (Dtype)1., bottom_diff);
        }

        // Update tdiff
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, N_, (Dtype)1.,
          top[0]->gpu_diff(), backprop_multiplier_.gpu_data(), (Dtype)0.,
          temp_tdiff_.mutable_gpu_diff());
        
        caffe_copy(temp_tdiff_.count(), temp_tdiff_.gpu_diff(), top[0]->mutable_gpu_diff());
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(SparseApproxLayer);

}  // namespace caffe
