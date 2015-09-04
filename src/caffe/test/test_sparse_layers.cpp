#include <cstring>
#include <vector>
#include <math.h>
#include <cfloat>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

/**
 * Warning: this test only works if A values do not cross 0
 * (i.e. gradient is not differentiable). Set inputs to 
 * positive only to make sure energy decreases.
 **/
template <typename TypeParam>
class SparseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  //create blob -> batch=3; channels=3 (RGB); pixels = 4 (2x2)
  SparseLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(4, 3, 2, 2)),
        blob_bottom_1_(new Blob<Dtype>(4, 10, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> dat_filler(filler_param);
    dat_filler.Fill(this->blob_bottom_0_);

    filler_param.set_min(0);
    filler_param.set_max(0);
    UniformFiller<Dtype> act_filler(filler_param);
    act_filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SparseLayerTest() { delete blob_bottom_0_; delete blob_bottom_1_; delete blob_top_; }

  Dtype compute_energy(shared_ptr<SparseUnitLayer<Dtype> > layer, LayerParameter layer_param){
    // E = 1/2 sum_p( (x[p] - sum_m(phi[m,p] * a[m]) - b[p])^2 ) + lambda_ * sum_m(a[m])
    Dtype lambda_    = layer_param.mutable_sparse_unit_param()->lambda();
    Dtype bias_term_ = layer_param.mutable_sparse_unit_param()->bias_term(); 
    const Blob<Dtype>* input    = this->blob_bottom_vec_[0];
    const Blob<Dtype>* activity = this->blob_top_vec_[0];
    const shared_ptr<Blob<Dtype> > weights = layer->blobs()[0];
    int batch_size   = this->blob_bottom_vec_[0]->shape(0); // M_
    int num_channels = this->blob_bottom_vec_[0]->shape(1);
    int num_pixelsH  = this->blob_bottom_vec_[0]->shape(2); // H
    int num_pixelsW  = this->blob_bottom_vec_[0]->shape(3); // W
    int num_elements = this->blob_bottom_vec_[1]->count(1); // M
    std::cout<<"\nnum_elements: "<<num_elements<<"\n";
    Dtype E = 0;
    for (int b=0; b < batch_size; ++b) {                    // batch
      Dtype residual_err = 0;
      Dtype a_sum = 0;
      for (int c=0; c < num_channels; ++c) {              // channel
        for (int h=0; h < num_pixelsH; ++h) {           // height
          for (int w=0; w < num_pixelsW; ++w) {       // width
            Dtype inner_term = 0;
            for (int m=0; m < num_elements; ++m) {  // elements
              if (c==0 && h==0 && w==0) {
                a_sum += std::abs(activity->cpu_data()[activity->offset(b, m)]);
              }
              inner_term += input->cpu_data()[input->offset(b, c, h, w)] - 
                  weights->cpu_data()[weights->offset(m, c*h*w+h*w+w)] * 
                  activity->cpu_data()[activity->offset(b, m)];
              if (bias_term_) {
                inner_term -= layer->blobs()[1]->cpu_data()[c*h*w+h*w+w];
              }
            }
            residual_err += inner_term * inner_term;
          }
        }
      }
      E += 0.5 * residual_err + lambda_ * a_sum;
      std::cout<<"a_sum: "<<a_sum<<"\n";
      std::cout<<"residual_error: "<<residual_err<<"\n";
    }
    return E;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseLayerTest, TestUnitSetUp) {

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  shared_ptr<SparseUnitLayer<Dtype> > layer(
      new SparseUnitLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->shape(0), 4);  // M_ -> Batch
  EXPECT_EQ(this->blob_top_->shape(1), 10); // N_ -> Elements
}

//TYPED_TEST(SparseLayerTest, TestUnitForward) {
//  typedef typename TypeParam::Dtype Dtype;
//  bool IS_VALID_CUDA = false;
//  #ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//  #endif
//  
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    SparseUnitParameter* sparse_unit_param =
//        layer_param.mutable_sparse_unit_param();
//
//    sparse_unit_param->set_lambda(0.1);
//    sparse_unit_param->set_eta(0.01);
//    sparse_unit_param->mutable_weight_filler()->set_type("uniform");
//    sparse_unit_param->mutable_weight_filler()->set_min(0);
//    sparse_unit_param->mutable_weight_filler()->set_max(0.1);
//    sparse_unit_param->set_bias_term(false);
//    sparse_unit_param->mutable_bias_filler()->set_type("uniform");
//    sparse_unit_param->mutable_bias_filler()->set_min(0);
//    sparse_unit_param->mutable_bias_filler()->set_max(0.5);
//
//    shared_ptr<SparseUnitLayer<Dtype> > layer(
//        new SparseUnitLayer<Dtype>(layer_param));
//
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//
//    Dtype E1 = this->compute_energy(layer,layer_param);
//    std::cout<<"Energy check E1: "<<E1<<"\n";
//
//    std::cout<<"\n--t=0--\n";
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    Dtype Eng = this->compute_energy(layer,layer_param);
//
//    std::cout<<"Energy check t=0 E="<<Eng<<"\n";
//    //CHECK_LE(Eng,E1)
//
//    vector<Blob<Dtype>*> new_blob_bottom_vec_;
//    for (int t = 1; t < 3; ++t) { 
//        std::cout<<"\n--t="<<t<<"--\n";
//        new_blob_bottom_vec_.clear();
//        new_blob_bottom_vec_.push_back(this->blob_bottom_vec_[0]);
//        new_blob_bottom_vec_.push_back(this->blob_top_vec_[0]);
//        layer->Forward(new_blob_bottom_vec_, this->blob_top_vec_);
//        //this->blob_bottom_vec_[1] = this->blob_top_vec_[0];
//        //layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//        Eng = this->compute_energy(layer,layer_param);
//        std::cout<<"Energy check t="<<t<<" E="<<Eng<<"\n";
//    }
//
//    //Dtype E2 = this->compute_energy(layer,layer_param);
//    //CHECK_LE(E2,E1);
//    CHECK_LE(Eng,E1);
//  }
//}

//TYPED_TEST(SparseLayerTest, TestUnitGradient) {
//}


//TYPED_TEST(SparseLayerTest, TestForward) {
//  typedef typename TypeParam::Dtype Dtype;
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//
//    //Set params
//    LayerParameter layer_param;
//    SparseUnitParameter* sparse_unit_param =
//        layer_param.mutable_sparse_unit_param();
//
//    sparse_unit_param->set_lambda(0.01);
//    sparse_unit_param->set_eta(0.1);
//    sparse_unit_param->set_bias_term(true);
//
//    // Set weights
//    sparse_unit_param->mutable_weight_filler()->set_type("uniform");
//    sparse_unit_param->mutable_weight_filler()->set_min(0);
//    sparse_unit_param->mutable_weight_filler()->set_max(1);
//
//    // Set bias
//    sparse_unit_param->mutable_bias_filler()->set_type("uniform");
//    sparse_unit_param->mutable_bias_filler()->set_min(0);
//    sparse_unit_param->mutable_bias_filler()->set_max(0.5);
//
//    // Create layer
//    shared_ptr<SparseUnitLayer<Dtype> > layer(
//        new SparseUnitLayer<Dtype>(layer_param));
//
//    // Setup
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//
//    // Compute E1
//    Dtype E1 = this->compute_energy(layer,layer_param);
//
//    // Forward Pass
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//
//    // Compute E2
//    Dtype E2 = this->compute_energy(layer,layer_param);
//
//    // Again with more iterations
//    layer->SetNumIterations(6,this->blob_bottom_vec_,this->blob_top_vec_);
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//
//    // Compute E3
//    Dtype E3 = this->compute_energy(layer,layer_param);
//
//    //Make sure E3 < E2 < E1
//    CHECK_LE(E2,E1);
//    CHECK_LE(E3,E2);
//    
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

//TYPED_TEST(SparseLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  bool IS_VALID_CUDA = false;
//
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//
//    LayerParameter layer_param;
//
//    SparseUnitParameter* sparse_unit_param =
//        layer_param.mutable_sparse_unit_param();
//
//    sparse_unit_param->set_num_iterations(10);
//    sparse_unit_param->set_num_elements(2);
//    sparse_unit_param->set_eta(0.2);
//    sparse_unit_param->set_lambda(0.1);
//
//    sparse_unit_param->mutable_weight_filler()->set_type("uniform");
//    sparse_unit_param->mutable_weight_filler()->set_min(0);
//    sparse_unit_param->mutable_weight_filler()->set_max(1);
//
//    sparse_unit_param->mutable_bias_filler()->set_type("uniform");
//    sparse_unit_param->mutable_bias_filler()->set_min(0);
//    sparse_unit_param->mutable_bias_filler()->set_max(1);
//
//    sparse_unit_param->set_bias_term(true);
//
//    SparseUnitLayer<Dtype> layer(layer_param);
//
//    Dtype stepsize  = 1e-2;
//    Dtype threshold = 1e-3;
//    GradientChecker<Dtype> checker(stepsize, threshold);
//
//    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//        this->blob_top_vec_);
//
//    sparse_unit_param->set_eta(2);
//
//    SparseUnitLayer<Dtype> layer2(layer_param);
//    GradientChecker<Dtype> checker2(stepsize, threshold);
//
//    checker2.CheckGradientExhaustive(&layer2, this->blob_bottom_vec_,
//        this->blob_top_vec_);
//
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

}  // namespace caffe
