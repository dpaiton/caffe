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
class SparseApproxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  //create blob -> batch=3; channels=3 (RGB); pixels = 4 (2x2)
  SparseApproxLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_top_(new Blob<Dtype>()) {

    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.5);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_); //top[0]
  }

  virtual ~SparseApproxLayerTest() { 
    delete blob_bottom_;
    delete blob_top_;
  }

  Dtype compute_energy(shared_ptr<SparseApproxLayer<Dtype> > layer, LayerParameter layer_param){
    // E = 1/2 sum_p( (x[p] - sum_m(phi[m,p] * a[m]) - b[p])^2 ) + lambda * sum_m(a[m])
    const Blob<Dtype>* input    = this->blob_bottom_vec_[0];
    const Blob<Dtype>* activity = this->blob_top_vec_[0]; // Sparse Approximation layer top
    const shared_ptr<Blob<Dtype> >  weights  = layer->blobs()[0];
    const shared_ptr<Blob<Dtype> >  bias     = layer->blobs()[1];
    int batch_size   = this->blob_bottom_vec_[0]->shape(0); // B
    int num_channels = this->blob_bottom_vec_[0]->shape(1);
    int num_pixelsH  = this->blob_bottom_vec_[0]->shape(2); // H
    int num_pixelsW  = this->blob_bottom_vec_[0]->shape(3); // W
    int num_elements = layer_param.mutable_sparse_approx_param()->num_elements(); // M
    Dtype lambda     = layer_param.mutable_sparse_approx_param()->lambda();
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
                                      activity->cpu_data()[activity->offset(b, m)] - 
                                      bias->cpu_data()[c*h*w+h*w+w];
                    }
                    residual_err += inner_term * inner_term;
                }
            }
        }
        E += 0.5 * residual_err + lambda * a_sum;
    }
    return E;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseApproxLayerTest, TestDtypesAndDevices);

//TYPED_TEST(SparseApproxLayerTest, TestSetUp) {
//
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//
//  SparseApproxParameter* sparse_approx_param =
//      layer_param.mutable_sparse_approx_param();
//
//  sparse_approx_param->set_num_iterations(4);
//  sparse_approx_param->set_num_elements(5);
//  
//  shared_ptr<SparseApproxLayer<Dtype> > layer(
//      new SparseApproxLayer<Dtype>(layer_param));
//
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//
//  EXPECT_EQ(this->blob_top_vec_[0]->shape(0), 3);  // B_ -> Batch
//  EXPECT_EQ(this->blob_top_vec_[0]->shape(1), 5);  // C_ -> num_elements
//}

//TYPED_TEST(SparseApproxLayerTest, TestForward) {
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
//    SparseApproxParameter* sparse_approx_param =
//        layer_param.mutable_sparse_approx_param();
//
//    sparse_approx_param->set_num_iterations(3);
//    sparse_approx_param->set_num_elements(5);
//    sparse_approx_param->set_lambda(0.1);
//    sparse_approx_param->set_eta(0.001);
//    sparse_approx_param->set_gamma(-FLT_MAX);
//    sparse_approx_param->set_bias_term(true);
//
//    // Set weights
//    sparse_approx_param->mutable_weight_filler()->set_type("uniform");
//    sparse_approx_param->mutable_weight_filler()->set_min(0);
//    sparse_approx_param->mutable_weight_filler()->set_max(1);
//
//    // Set bias
//    sparse_approx_param->mutable_bias_filler()->set_type("uniform");
//    sparse_approx_param->mutable_bias_filler()->set_min(0);
//    sparse_approx_param->mutable_bias_filler()->set_max(0.5);
//
//    // Create layer
//    shared_ptr<SparseApproxLayer<Dtype> > layer(
//        new SparseApproxLayer<Dtype>(layer_param));
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

TYPED_TEST(SparseApproxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;

#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif

  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

    LayerParameter layer_param;

    SparseApproxParameter* sparse_approx_param =
              layer_param.mutable_sparse_approx_param();

    sparse_approx_param->set_num_iterations(3);
    sparse_approx_param->set_num_elements(1);
    sparse_approx_param->set_eta(2);
    sparse_approx_param->set_lambda(0);
    //sparse_approx_param->set_gamma(-FLT_MAX);

    sparse_approx_param->set_bias_term(true);

    // Set weights
    sparse_approx_param->mutable_weight_filler()->set_type("uniform");
    sparse_approx_param->mutable_weight_filler()->set_min(0.01);
    sparse_approx_param->mutable_weight_filler()->set_max(0.1);

    // Set bias
    sparse_approx_param->mutable_bias_filler()->set_type("uniform");
    sparse_approx_param->mutable_bias_filler()->set_min(0);
    sparse_approx_param->mutable_bias_filler()->set_max(0);

    SparseApproxLayer<Dtype> layer(layer_param);

    Dtype stepsize  = 1e-2;
    Dtype threshold = 1e-3;
    GradientChecker<Dtype> checker(stepsize, threshold);

    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
         this->blob_top_vec_);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
