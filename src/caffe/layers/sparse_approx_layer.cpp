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

  // Make sure bottom.num_axes() == 4? Maybe this is guaranteed?

  //const int num_iterations_ = 
  //    this->layer_param_.sparse_approx_param().num_iterations();

  bias_term_ = this->layer_param_.sparse_approx_param().bias_term();

  //Find variables B_, L_, M_
  B_ = bottom[0]->shape(0);
  L_ = bottom[0]->shape(1); // because C is singleton dimension, you don't index over it
  M_ = this->layer_param_.sparse_approx_param().num_elements();

  //TODO:Verify that blob dimensions are correct for math

  // Allocate weights
  if (bias_term_) {
      this->blobs_.resize(2); // phi & bias
  } else {
      this->blobs_.resize(1); // only one stored param, phi
  }

  // Intialize weights
  // TODO: Should this have been BxCxLxM?
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
      vector<int> bias_shape(1,M_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
  } 
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Set size of biased_input_ (MxB)
  vector<int> biased_input_shape(2);
  biased_input_shape[0] = B_;
  biased_input_shape[1] = M_;
  biased_input_.Reshape(biased_input_shape);

  // Set size of activity history matrix (MxT, where T = num_iterations
  vector<int> activity_history_shape(2);
  activity_history_shape[0] = M_;
  activity_history_shape[1] = num_iterations_;
  activity_history_.Reshape(activity_history_shape);
  caffe_set(activity_history_.count(), (Dtype)0., activity_history_.mutable_cpu_data());

  // Set size of competition matrix (G is of dim MxM)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = M_;
  competition_matrix_shape[1] = M_;
  competition_matrix_.Reshape(competition_matrix_shape);

  // Set size of top blob (MxB)
  top[0]->ReshapeLike(biased_input_);
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  //temp_0_ holds b values
  Blob<Dtype> temp_0_;
  temp_0_.ReshapeLike(biased_input_);

  Dtype eta    = this->layer_param_.sparse_approx_param().eta();
  Dtype lambda = this->layer_param_.sparse_approx_param().lambda();

  const Dtype* bottom_data = bottom[0]->cpu_data(); // x

  caffe_sub(bottom[0]->count(),bottom_data,this->blobs_[1]->cpu_data(),biased_input_.mutable_cpu_data()); // x-b

  const Dtype* weights = this->blobs_[0]->cpu_data(); // phi

  //Dtype* top_data = top[0]->mutable_cpu_data(); // f(a)

  // u(t+1) = (1-tau) u(t) + tau [x**T phi - u(t) phi**T phi T(u(t)-lambda) ]
  // a(t+1) = u(t+1) T(u(t+1)-lambda)
  //
  // f(a) = a + eta [ x**T phi - a phi**T phi - lambda sgn(a)]
  //
  // b = gemm(x**T,phi)    (BxL) * (LxM) = (BxM)  ->  top
  // g = gemm(phi**T,phi)  (MxL) * (LxM) = (MxM)  ->  competition_matrix_
  // ag = gemm(a,g)        (BxM) * (MxM) = (BxM)  ->  temp_0_
  // b_ga = sub(b, ag)
  // f(a) = add(a, eta * add(b_ga, lambda sgn(a)))
  // 
  // BxCxLxM -- batch x feature (color) x pixels x elements
  // phi -- LxM  //  u,a -- BxM  //  x -- LxB  //  


  // First iteration
  // g
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, M_, L_,
          (Dtype)1., weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  // b
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, M_, L_,
          (Dtype)1., biased_input_.cpu_data(), weights, (Dtype)0.,
          temp_0_.mutable_cpu_data());
  caffe_copy(top[0]->count(),temp_0_.cpu_data(),activity_history_.mutable_cpu_data());

  // ag = 0
  
  // f(a)
  caffe_scal(top[0]->count(),eta,activity_history_.mutable_cpu_data());

  // Next iterations
  for (int iteration = 1; iteration < num_iterations_; iteration++) {
      // Set up pointers
      Dtype* mutable_a_current     = activity_history_.mutable_cpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_current = activity_history_.cpu_data() + activity_history_.offset(iteration);
      const Dtype* const_a_past    = activity_history_.cpu_data() + activity_history_.offset(iteration-1);

      // Add b value to output, store in current history slot
      caffe_add(top[0]->count(), temp_0_.cpu_data(), const_a_current, mutable_a_current);

      // Compute b - a[iteration-1] g, store in current history slot
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, B_, M_,
              (Dtype)-1., const_a_past, competition_matrix_.cpu_data(), (Dtype)1.,
              mutable_a_current);

      // Compute sign on previous a's, store in top temporarily
      caffe_cpu_sign(top[0]->count(), const_a_past, top[0]->mutable_cpu_data());
      // lambda * sign(a), store in top temporarily
      caffe_scal(top[0]->count(),lambda,top[0]->mutable_cpu_data()); 

      // Store [ ... ] in current history slot
      caffe_sub(top[0]->count(), const_a_current, top[0]->cpu_data(),
              mutable_a_current);
      // eta * [ ... ], store in current hist slot
      caffe_scal(top[0]->count(), eta, mutable_a_current);

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

}

//TODO: Look into these

//#ifdef CPU_ONLY
//STUB_GPU(SparseApproxLayer);
//#endif

INSTANTIATE_CLASS(SparseApproxLayer);
REGISTER_LAYER_CLASS(SparseApprox);

}  // namespace caffe
