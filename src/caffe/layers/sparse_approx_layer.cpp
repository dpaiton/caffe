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

  //Find variables B_, L_, M_
  B_ = bottom[0]->shape(0);
  L_ = bottom[0]->shape(2);
  M_ = this->layer_param_.sparse_approx_param().num_elements();

  //TODO:Verify that blob dimensions are correct for math

  // Allocate weights
  this->blobs_.resize(1); // only one stored param, phi

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

}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Set size of activities_ (MxB)
  vector<int> activities_shape(2);
  activities_shape[0] = B_;
  activities_shape[1] = M_;
  activities_.Reshape(activities_shape);

  // Set size of activity history matrix (MxT, where T = num_iterations
  vector<int> activity_history_shape(2);
  activity_history_shape[0] = M_;
  activity_history_shape[1] = num_iterations_;
  activity_history_.Reshape(activity_history_shape);

  // Set size of competition matrix (G is of dim MxM)
  vector<int> competition_matrix_shape(2);
  competition_matrix_shape[0] = M_;
  competition_matrix_shape[1] = M_;
  competition_matrix_.Reshape(competition_matrix_shape);

  // Set size of top blob (MxB) (activity_values)
  vector<int> top_shape(2);
  top_shape[0] = M_;
  top_shape[1] = B_;
  top[0]->Reshape(top_shape);

}

template <typename Dtype>
void SparseApproxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Blob<Dtype> temp_0_, temp_1_, temp_diff_;
  vector<int> temp_shape(2);
  temp_shape[0] = B_;
  temp_shape[1] = M_;
  temp_0_.Reshape(temp_shape);
  temp_1_.Reshape(temp_shape);
  temp_diff_.Reshape(temp_shape);

  //float lambda = this->layer_param_.sparse_approx_param().lambda();
  //float eta    = this->layer_param_.sparse_approx_param().eta();

  //TODO: Not exactly sure how this is working... 
  const Dtype* bottom_data = bottom[0]->cpu_data(); // x

  const Dtype* weights = this->blobs_[0]->cpu_data(); // phi

  //Dtype* top_data = top[0]->mutable_cpu_data(); // f(a)


  // f(a) = a + eta [ x**T phi - a phi**T phi - lambda sgn(a)]
  //
  // b = gemm(x**T,phi)    (BxL) * (LxM) = (BxM)  ->  temp_0_
  // g = gemm(phi**T,phi)  (MxL) * (LxM) = (MxM)  ->  competition_matrix_
  // ag = gemm(a,g)        (BxM) * (MxM) = (BxM)  ->  temp_1_
  // b_ga = sub(b, ga)
  // f(a) = add(a, eta * add(b_ga, lambda * sgn(a)))
  // 
  // BxCxLxM -- batch x feature (color) x pixels x elements
  // phi -- LxM  //  u,a -- BxM  //  x -- LxB  //  


  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, B_, M_, L_,
          (Dtype)1., bottom_data, weights, (Dtype)0., temp_0_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, M_, L_,
          (Dtype)1., weights, weights, (Dtype)0., competition_matrix_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, B_, M_,
          (Dtype)1., activities_.cpu_data(), competition_matrix_.cpu_data(), (Dtype)0.,
          temp_1_.mutable_cpu_data());

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
