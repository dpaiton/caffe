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
class GramianLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  //create blob -> numBatch=3; numElements=5; numPixels = 4 (2x2)
  GramianLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 5, 2, 2)),
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

  virtual ~GramianLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GramianLayerTest, TestDtypesAndDevices);

TYPED_TEST(GramianLayerTest, TestSetUp) {

  int numBatch    = 2;
  int numElements = 5;

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  shared_ptr<GramianLayer<Dtype> > layer(
      new GramianLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->shape(0), numBatch);  // M_ -> Batch
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1), numElements*numElements); // K_*K_ 
}

TYPED_TEST(GramianLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

    int numBatch    = 2;
    int numElements = 5;
    int numPixelsW  = 2;
    int numPixelsH  = 2;

    //Set params
    LayerParameter layer_param;

    // Create layer
    shared_ptr<GramianLayer<Dtype> > layer(
        new GramianLayer<Dtype>(layer_param));

    // Setup
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    // Forward Pass
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype* top_data = this->blob_top_vec_[0]->cpu_data();
    const Dtype* bot_data = this->blob_bottom_vec_[0]->cpu_data();
    for (int b = 0; b < numBatch; ++b) {
        for (int i = 0; i < numElements; ++i) {
            for (int j = 0; j < numElements; ++j) {
                Dtype sum = 0;
                for (int w = 0; w < numPixelsW; ++w) {
                    for (int h = 0; h < numPixelsH; ++h) {
                        Dtype act1 = bot_data[this->blob_bottom_vec_[0]->offset(b,i,w,h)];
                        Dtype act2 = bot_data[this->blob_bottom_vec_[0]->offset(b,j,w,h)];
                        sum += act1 * act2;
                    }
                }
                Dtype out = top_data[this->blob_top_vec_[0]->offset(b, i*numElements+j)];
                ASSERT_NEAR(out,sum,1e-6);
            }
        }
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GramianLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;

#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif

  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

    LayerParameter layer_param;

    GramianLayer<Dtype> layer(layer_param);

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
