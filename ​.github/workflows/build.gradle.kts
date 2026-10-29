plugins {
id("com.android.application")
id("org.jetbrains.kotlin.android")
}
​android {
namespace = "com.eyeoftheworld.app"
compileSdk = 34
​defaultConfig {
applicationId = "com.eyeoftheworld.app"
minSdk = 24
targetSdk = 34
versionCode = 1
versionName = "1.0.0"
​testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
}
​buildTypes {
release {
isMinifyEnabled = false
proguardFiles(
getDefaultProguardFile("proguard-android-optimize.txt"),
"proguard-rules.pro"
)
}
}
​compileOptions {
sourceCompatibility = JavaVersion.VERSION_17
targetCompatibility = JavaVersion.VERSION_17
}
​kotlinOptions {
jvmTarget = "17"
}
​// عدم ضغط ملف نموذج الذكاء الاصطناعي داخل التطبيق
aaptOptions {
noCompress("tflite")
}
}
​dependencies {
// مكتبات أندرويد والواجهة
implementation("androidx.core:core-ktx:1.12.0")
implementation("androidx.appcompat:appcompat:1.6.1")
implementation("com.google.android.material:material:1.11.0")
​// مكتبات الكاميرا CameraX
val cameraVersion = "1.3.1"
implementation("androidx.camera:camera-core:$cameraVersion")
implementation("androidx.camera:camera-camera2:$cameraVersion")
implementation("androidx.camera:camera-lifecycle:$cameraVersion")
implementation("androidx.camera:camera-view:$cameraVersion")
​// محرك الذكاء الاصطناعي LiteRT / TensorFlow Lite
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")
}
