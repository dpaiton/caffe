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

template <typename TypeParam>
class BiasLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  //create blob -> numBatch=3; numChannels=3; numPixels = 4 (2x2)
  BiasLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 3, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~BiasLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BiasLayerTest, TestDtypesAndDevices);

TYPED_TEST(BiasLayerTest, TestSetUp) {

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  shared_ptr<BiasLayer<Dtype> > layer(
      new BiasLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->shape(), this->blob_bottom_vec_[0]->shape());
}

TYPED_TEST(BiasLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

    //Set params
    LayerParameter layer_param;
    BiasParameter* bias_param =
        layer_param.mutable_bias_param();
    bias_param->mutable_bias_filler()->set_type("uniform");
    bias_param->mutable_bias_filler()->set_min(1);
    bias_param->mutable_bias_filler()->set_max(2);

    // Create layer
    shared_ptr<BiasLayer<Dtype> > layer(
        new BiasLayer<Dtype>(layer_param));

    // Setup & forward pass
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype* input = this->blob_bottom_vec_[0]->cpu_data();
    const Dtype* bias = layer->blobs()[0]->cpu_data();
    const Dtype* output = this->blob_top_vec_[0]->cpu_data();
    const int count = this->blob_top_vec_[0]->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(output[i], input[i]+bias[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(BiasLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;

#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif

  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

    // Set params
    LayerParameter layer_param;
    BiasParameter* bias_param =
        layer_param.mutable_bias_param();
    bias_param->mutable_bias_filler()->set_type("uniform");
    bias_param->mutable_bias_filler()->set_min(1);
    bias_param->mutable_bias_filler()->set_max(2);

    // Create layer
    BiasLayer<Dtype> layer(layer_param);

    // Check gradient
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
