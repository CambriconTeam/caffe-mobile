#include <jni.h>

#include "caffe_classification.hpp"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_setBlasThreadNum(JNIEnv *env, jobject instance,
                                                            jint numThreads) {
  openblas_set_num_threads(numThreads);
}

JNIEXPORT void JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_hello(JNIEnv *env,jobject instance) {
  LOG(ERROR) << "hello world";
}

JNIEXPORT jboolean JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_loadModel(JNIEnv *env, jobject instance,
                                                     jstring modelPath_, jstring weightPath_) {
    jboolean ret = true;
    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);

    if (caffe::CaffeClassification::get(modelPath, weightPath) == NULL) {
        ret = false;
    }

    env->ReleaseStringUTFChars(modelPath_, modelPath);
    env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;
}

JNIEXPORT jfloatArray JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_predict(JNIEnv *env, jobject instance,
                                                   jbyteArray jrgba, jint jchannels, jfloatArray jmean) {
  uint8_t *rgba = NULL;
  // Get matrix pointer
  if (NULL != jrgba) {
    rgba = (uint8_t *)env->GetByteArrayElements(jrgba, 0);
  } else {
    LOG(ERROR) << "caffe-jni predict(): invalid args: jrgba(NULL)";
    return NULL;
  }
  std::vector<float> mean;
  if (NULL != jmean) {
    float * mean_arr = (float *)env->GetFloatArrayElements(jmean, 0);
    int mean_size = env->GetArrayLength(jmean);
    mean.assign(mean_arr, mean_arr+mean_size);
  } else {
    LOG(INFO) << "caffe-jni predict(): args: jmean(NULL)";
  }
  // Predict
  caffe::CaffeClassification *caffe_classification = caffe::CaffeClassification::get();
  if (NULL == caffe_classification) {
    LOG(ERROR) << "caffe-jni predict(): CaffeClassification failed to initialize";
    return NULL;  // not initialized
  }
  int rgba_len = env->GetArrayLength(jrgba);
  if (rgba_len != jchannels * caffe_classification->input_width() * caffe_classification->input_height()) {
    LOG(WARNING) << "caffe-jni predict(): invalid rgba length(" << rgba_len << ") expect(" <<
                    jchannels * caffe_classification->input_width() * caffe_classification->input_height() << ")";
    return NULL;  // not initialized
  }
  std::vector<float> predict;
  if (!caffe_classification->predictImage(rgba, jchannels, mean, predict)) {
    LOG(WARNING) << "caffe-jni predict(): CaffeClassification failed to predict";
    return NULL; // predict error
  }
  // Handle result
  jfloatArray result = env->NewFloatArray(predict.size());
  if (result == NULL) {
    return NULL; // out of memory error thrown
  }
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, predict.size(), predict.data());
  return result;
}

JNIEXPORT jint JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_inputChannels(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeClassification *caffe_classification = caffe::CaffeClassification::get();
  if (NULL == caffe_classification) {
      return -1;  // not initialized
  }
  return caffe_classification->input_channels();
}

JNIEXPORT jint JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_inputWidth(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeClassification *caffe_classification = caffe::CaffeClassification::get();
  if (NULL == caffe_classification) {
      return -1;  // not initialized
  }
  return caffe_classification->input_width();
}

JNIEXPORT jint JNICALL
Java_com_cambricon_productdisplay_caffenative_CaffeClassification_inputHeight(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeClassification *caffe_classification = caffe::CaffeClassification::get();
  if (NULL == caffe_classification) {
    return -1;  // not initialized
  }
  return caffe_classification->input_height();
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }
  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
