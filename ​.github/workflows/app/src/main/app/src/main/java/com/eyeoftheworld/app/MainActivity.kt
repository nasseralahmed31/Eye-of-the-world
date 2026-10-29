// ============================================================================
// عين العالم | Eye of the World - Android Application Source Code
// Modern Kotlin, CameraX, LiteRT (TFLite), Custom Canvas Glow, & TextToSpeech
// ============================================================================

package com.eyeoftheworld.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

// ============================================================================
// 1. DATA MODELS & UTILS
// ============================================================================

data class TrackedObject(
    val id: Int,
    var labelAr: String,
    var labelEn: String,
    var confidence: Float,
    var maskPath: Path,
    var boundingRect: RectF,
    var lastSeenTimestamp: Long = System.currentTimeMillis()
)

data class InferenceResult(
    val trackedObjects: List<TrackedObject>,
    val processTimeMs: Long,
    val frameWidth: Int,
    val frameHeight: Int
)

object LabelUtils {
    val LABELS_AR = arrayOf(
        "خلفية", "طائرة", "دراجة هوائية", "طائر", "قارب",
        "زجاجة", "حافلة", "سيارة", "قطة", "كرسي",
        "بقرة", "طاولة طعام", "كلب", "حصان", "دراجة نارية",
        "شخص", "نبتة منزلية", "خروف", "أريكة", "قطار", "شاشة / تلفاز"
    )

    val LABELS_EN = arrayOf(
        "Background", "Aeroplane", "Bicycle", "Bird", "Boat",
        "Bottle", "Bus", "Car", "Cat", "Chair",
        "Cow", "Dining Table", "Dog", "Horse", "Motorbike",
        "Person", "Potted Plant", "Sheep", "Sofa", "Train", "TV/Monitor"
    )

    fun getLabelAr(index: Int): String = if (index in LABELS_AR.indices) LABELS_AR[index] else "كائن مجهول"
    fun getLabelEn(index: Int): String = if (index in LABELS_EN.indices) LABELS_EN[index] else "Unknown"
}

// ============================================================================
// 2. SEGMENTATION ENGINE (LiteRT / TensorFlow Lite)
// ============================================================================

class SegmenterEngine(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val inputSize = 257 // Standard DeepLabV3 Input Size
    private val numClasses = 21
    private val imgData: ByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    private val segmentBits = Array(inputSize) { IntArray(inputSize) }
    private var nextObjectId = 1

    init {
        initInterpreter()
    }

    private fun initInterpreter() {
        try {
            val options = Interpreter.Options()
            val compatList = CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(4)
            }

            val modelBuffer = loadModelFile("deeplabv3_257.tflite")
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("SegmenterEngine", "Error initializing LiteRT model: ${e.localizedMessage}")
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun analyzeBitmap(bitmap: Bitmap, rotationDegrees: Int): InferenceResult {
        val startTime = SystemClock.uptimeMillis()
        val objects = mutableListOf<TrackedObject>()

        if (interpreter == null) {
            return InferenceResult(emptyList(), SystemClock.uptimeMillis() - startTime, bitmap.width, bitmap.height)
        }

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        convertBitmapToByteBuffer(resizedBitmap)

        val outputBuffer = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(numClasses) } } }
        interpreter?.run(imgData, outputBuffer)

        val detectedClasses = HashSet<Int>()
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                var maxVal = -Float.MAX_VALUE
                var maxClass = 0
                for (c in 0 until numClasses) {
                    val value = outputBuffer[0][y][x][c]
                    if (value > maxVal) {
                        maxVal = value
                        maxClass = c
                    }
                }
                segmentBits[y][x] = maxClass
                if (maxClass > 0) detectedClasses.add(maxClass)
            }
        }

        for (cls in detectedClasses) {
            val maskPath = Path()
            var minX = inputSize
            var minY = inputSize
            var maxX = 0
            var maxY = 0
            var pixelCount = 0

            for (y in 0 until inputSize step 2) {
                for (x in 0 until inputSize step 2) {
                    if (segmentBits[y][x] == cls) {
                        pixelCount++
                        if (x < minX) minX = x
                        if (x > maxX) maxX = x
                        if (y < minY) minY = y
                        if (y > maxY) maxY = y

                        maskPath.addRect(
                            x.toFloat(), y.toFloat(),
                            (x + 2).toFloat(), (y + 2).toFloat(),
                            Path.Direction.CW
                        )
                    }
                }
            }

            if (pixelCount > 100) {
                val rectF = RectF(minX.toFloat(), minY.toFloat(), maxX.toFloat(), maxY.toFloat())
                val confidence = min(0.99f, 0.65f + (pixelCount / (inputSize * inputSize.toFloat())))

                objects.add(
                    TrackedObject(
                        id = nextObjectId++,
                        labelAr = LabelUtils.getLabelAr(cls),
                        labelEn = LabelUtils.getLabelEn(cls),
                        confidence = confidence,
                        maskPath = maskPath,
                        boundingRect = rectF
                    )
                )
            }
        }

        val elapsedTime = SystemClock.uptimeMillis() - startTime
        return InferenceResult(objects, elapsedTime, inputSize, inputSize)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData.rewind()
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val valPixel = intValues[pixel++]
                imgData.putFloat((((valPixel shr 16) and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat((((valPixel shr 8) and 0xFF) - 127.5f) / 127.5f)
                imgData.putFloat(((valPixel and 0xFF) - 127.5f) / 127.5f)
            }
        }
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}

// ============================================================================
// 3. SCI-FI HUD OVERLAY (Neon Glow Overlay)
// ============================================================================

class SciFiOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var currentResult: InferenceResult? = null

    private val outerGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 14f
        color = Color.parseColor("#00E5FF")
        maskFilter = BlurMaskFilter(18f, BlurMaskFilter.Blur.NORMAL)
        alpha = 180
    }

    private val innerBrightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.parseColor("#E0FFFF")
    }

    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CC0A1128")
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00F0FF")
        textSize = 38f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    }

    private val cornerBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#00E5FF")
    }

    init {
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    fun updateResults(result: InferenceResult) {
        this.currentResult = result
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val result = currentResult ?: return
        if (result.trackedObjects.isEmpty()) return

        val scaleX = width.toFloat() / result.frameWidth
        val scaleY = height.toFloat() / result.frameHeight

        for (obj in result.trackedObjects) {
            canvas.save()
            canvas.scale(scaleX, scaleY)

            canvas.drawPath(obj.maskPath, outerGlowPaint)
            canvas.drawPath(obj.maskPath, innerBrightPaint)

            canvas.restore()

            val mappedRect = RectF(
                obj.boundingRect.left * scaleX,
                obj.boundingRect.top * scaleY,
                obj.boundingRect.right * scaleX,
                obj.boundingRect.bottom * scaleY
            )
            drawSciFiCorners(canvas, mappedRect)

            val labelText = "${obj.labelAr} | ${(obj.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(labelText)
            val badgeHeight = 60f

            val badgeLeft = max(10f, mappedRect.left)
            val badgeTop = max(10f, mappedRect.top - badgeHeight - 10f)

            canvas.drawRoundRect(
                badgeLeft, badgeTop, badgeLeft + textWidth + 30f, badgeTop + badgeHeight,
                12f, 12f, textBgPaint
            )
            canvas.drawRoundRect(
                badgeLeft, badgeTop, badgeLeft + textWidth + 30f, badgeTop + badgeHeight,
                12f, 12f, cornerBorderPaint
            )

            canvas.drawText(labelText, badgeLeft + 15f, badgeTop + 40f, textPaint)
        }
    }

    private fun drawSciFiCorners(canvas: Canvas, rect: RectF) {
        val len = min(rect.width(), rect.height()) * 0.15f
        canvas.drawLine(rect.left, rect.top, rect.left + len, rect.top, cornerBorderPaint)
        canvas.drawLine(rect.left, rect.top, rect.left, rect.top + len, cornerBorderPaint)
        canvas.drawLine(rect.right - len, rect.top, rect.right, rect.top, cornerBorderPaint)
        canvas.drawLine(rect.right, rect.top, rect.right, rect.top + len, cornerBorderPaint)
        canvas.drawLine(rect.left, rect.bottom, rect.left + len, rect.bottom, cornerBorderPaint)
        canvas.drawLine(rect.left, rect.bottom - len, rect.left, rect.bottom, cornerBorderPaint)
        canvas.drawLine(rect.right - len, rect.bottom, rect.right, rect.bottom, cornerBorderPaint)
        canvas.drawLine(rect.right, rect.bottom - len, rect.right, rect.bottom, cornerBorderPaint)
    }
}

// ============================================================================
// 4. MAIN ACTIVITY & UI
// ============================================================================

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var cameraPreview: PreviewView
    private lateinit var overlayView: SciFiOverlayView
    private lateinit var statusTv: TextView
    private lateinit var fpsTv: TextView
    private lateinit var btnFreeze: Button
    private lateinit var btnToggleCam: Button
    private lateinit var btnMuteTts: ImageButton

    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var isFrozen = false
    private var isMuted = false

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var segmenterEngine: SegmenterEngine
    private var tts: TextToSpeech? = null
    private var lastSpokenTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val rootLayout = RelativeLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        cameraPreview = PreviewView(this).apply {
            layoutParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.MATCH_PARENT
            )
        }
        rootLayout.addView(cameraPreview)

        overlayView = SciFiOverlayView(this).apply {
            layoutParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.MATCH_PARENT
            )
        }
        rootLayout.addView(overlayView)

        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(30, 40, 30, 20)
            setBackgroundColor(Color.parseColor("#99050B14"))
            layoutParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT
            )
        }

        statusTv = TextView(this).apply {
            text = "عين العالم: النظام متصل وجاهز"
            setTextColor(Color.parseColor("#00E5FF"))
            textSize = 16f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        }

        fpsTv = TextView(this).apply {
            text = "زمن الاستدلال: --ms | معدل الإطارات: --"
            setTextColor(Color.parseColor("#7C4DFF"))
            textSize = 12f
        }

        headerLayout.addView(statusTv)
        headerLayout.addView(fpsTv)
        rootLayout.addView(headerLayout)

        val bottomPanel = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#B3050B14"))
            setPadding(20, 20, 20, 30)
            val lp = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.MATCH_PARENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                addRule(RelativeLayout.ALIGN_PARENT_BOTTOM)
            }
            layoutParams = lp
        }

        btnFreeze = Button(this).apply {
            text = "تجميد"
            setBackgroundColor(Color.parseColor("#00B0FF"))
            setTextColor(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                setMargins(10, 0, 10, 0)
            }
            setOnClickListener {
                isFrozen = !isFrozen
                text = if (isFrozen) "استئناف" else "تجميد"
            }
        }

        btnToggleCam = Button(this).apply {
            text = "كاميرا"
            setBackgroundColor(Color.parseColor("#00E676"))
            setTextColor(Color.BLACK)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                setMargins(10, 0, 10, 0)
            }
            setOnClickListener {
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                    CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
                bindCameraUseCases()
            }
        }

        btnMuteTts = ImageButton(this).apply {
            setImageResource(android.R.drawable.ic_lock_silent_mode_off)
            setColorFilter(Color.CYAN)
            setBackgroundColor(Color.TRANSPARENT)
            layoutParams = LinearLayout.LayoutParams(120, 120)
            setOnClickListener {
                isMuted = !isMuted
                setColorFilter(if (isMuted) Color.RED else Color.CYAN)
            }
        }

        bottomPanel.addView(btnFreeze)
        bottomPanel.addView(btnToggleCam)
        bottomPanel.addView(btnMuteTts)
        rootLayout.addView(bottomPanel)

        setContentView(rootLayout)

        cameraExecutor = Executors.newSingleThreadExecutor()
        segmenterEngine = SegmenterEngine(this)
        tts = TextToSpeech(this, this)

        checkPermissionsAndStart()
    }

    private fun checkPermissionsAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(cameraPreview.surfaceProvider)
        }

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetResolution(Size(640, 480))
            .build()

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            if (!isFrozen) {
                val bitmap = imageProxy.toBitmap()
                if (bitmap != null) {
                    val result = segmenterEngine.analyzeBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)

                    runOnUiThread {
                        overlayView.updateResults(result)
                        statusTv.text = "الأشياء المكتشفة: ${result.trackedObjects.size}"
                        fpsTv.text = "زمن المعالجة: ${result.processTimeMs} ms | الوضع: مباشر"

                        if (!isMuted && result.trackedObjects.isNotEmpty()) {
                            val now = System.currentTimeMillis()
                            if (now - lastSpokenTime > 4000L) {
                                val topTarget = result.trackedObjects.maxByOrNull { it.confidence }
                                topTarget?.let {
                                    tts?.speak("تم رصد ${it.labelAr}", TextToSpeech.QUEUE_FLUSH, null, "EYE_TTS")
                                    lastSpokenTime = now
                                }
                            }
                        }
                    }
                }
            }
            imageProxy.close()
        }

        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        try {
            provider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
        } catch (e: Exception) {
            Log.e("MainActivity", "Camera binding failed", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale("ar")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        segmenterEngine.close()
        tts?.stop()
        tts?.shutdown()
    }
}
