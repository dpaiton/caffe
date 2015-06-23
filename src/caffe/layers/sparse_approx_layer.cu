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

  // Subtract bias values from input
  for (int batch=0; batch < bottom[0]->shape(0); batch++) { // same bias is applied to each batch item
      caffe_gpu_sub(L_, bottom[0]->gpu_data() + bottom[0]->offset(batch),
              this->blobs_[1]->gpu_data(),
              biased_input_.mutable_gpu_data() + biased_input_.offset(batch)); // x = x - bias
  }

  const Dtype* weights = this->blobs_[0]->gpu_data(); // phi

  //temp_0 holds ext values
  Blob<Dtype> temp_0;
  temp_0.Reshape(top[0]->shape());

  // First iteration
  // g
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, M_, L_,
          (Dtype)1., weights, weights, (Dtype)0., competition_matrix_.mutable_gpu_data());

  // ext 
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, L_,
          (Dtype)1., biased_input_.gpu_data(), weights, (Dtype)0.,
          temp_0.mutable_gpu_data());

  caffe_copy(top[0]->count(), temp_0.gpu_data(), activity_history_.mutable_gpu_data());
  
  // f(a)
  caffe_gpu_scal(top[0]->count(), eta_, activity_history_.mutable_gpu_data());

  // Next iterations
  for (int iteration = 1; iteration < num_iterations_; iteration++) {
      // Set up pointers
      Dtype* mutable_a_current     = activity_history_.mutable_gpu_data() + activity_history_.offset(iteration);
      Dtype* mutable_a_past        = activity_history_.mutable_gpu_data() + activity_history_.offset(iteration-1);
      const Dtype* const_a_current = activity_history_.gpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_past    = activity_history_.gpu_data() + activity_history_.offset(iteration-1);

      //// Threshold previous activities
      //// Currently implements rectified soft threshold.
      // Look into threshold layer cu file.
      // can't dereference GPU data pointer, have to use CPU pointer
      // if it is best to do this in the CPU then you'll have to store flags
      //for (int i = 0; i < M_; i++) {
      //    if (const_a_past[i] < gamma_) {
      //        caffe_gpu_set(1, (Dtype)0., mutable_a_past + i);
      //    }
      //}

      // Add ext value to output, store in current history slot
      caffe_gpu_add(top[0]->count(), temp_0.gpu_data(), const_a_current, mutable_a_current);

      // Compute ext - a[iteration-1] g, store in current history slot
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, M_, M_,
              (Dtype)-1., const_a_past, competition_matrix_.gpu_data(), (Dtype)1.,
              mutable_a_current);

     // Compute sign on previous a's, store in top temporarily
     caffe_gpu_sign(top[0]->count(), const_a_past, top[0]->mutable_gpu_data());

     // lambda_ * sign(a), store in top temporarily
     caffe_gpu_scal(top[0]->count(), lambda_, top[0]->mutable_gpu_data()); 

     // Store [ ... ] in current history slot
     caffe_gpu_sub(top[0]->count(), const_a_current, top[0]->gpu_data(), 
             mutable_a_current);

     // eta_ * [ ... ], store in current hist slot
     caffe_gpu_scal(top[0]->count(), eta_, mutable_a_current);

     // Add previous activities to current acticities, store in current slot
     caffe_gpu_add(top[0]->count(), const_a_past, const_a_current, mutable_a_current);
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
    //Do stuff
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
