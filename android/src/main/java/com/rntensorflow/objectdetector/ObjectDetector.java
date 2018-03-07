package com.rntensorflow.objectdetector;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import com.facebook.react.bridge.*;
import com.rntensorflow.ResourceManager;

import java.util.List;

public class ObjectDetector {
    private static final int INPUT_SIZE = 300;

    private Classifier detector;
    private ResourceManager resourceManager;

    private ObjectDetector(ResourceManager resourceManager, Classifier detector) {
        this.detector = detector;
        this.resourceManager = resourceManager;
    }

    public static ObjectDetector create(ReactContext reactContext, String modelPath, String labelPath) throws Exception {
        ResourceManager resourceManager = new ResourceManager(reactContext);
        Classifier detector = TensorFlowObjectDetectionAPIModel.create(
                resourceManager, modelPath, labelPath, INPUT_SIZE);
        return new ObjectDetector(resourceManager, detector);
    }

    public WritableArray detect(String image) {
        Bitmap bitmap = imagePathToBitmap(image);
        List<Classifier.Recognition> recognitions = detector.recognizeImage(bitmap);
        return convertRecognitionToWritableArray(recognitions);
    }

    private Bitmap imagePathToBitmap(String image) {
        Bitmap bitmapRaw = loadImage(resourceManager.loadResource(image));
        Bitmap bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Matrix matrix = createMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(), INPUT_SIZE, INPUT_SIZE);
        final Canvas canvas = new Canvas(bitmap);
        canvas.drawBitmap(bitmapRaw, matrix, null);
        return bitmap;
    }

    private Bitmap loadImage(byte[] image) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeByteArray(image, 0, image.length);
    }

    private Matrix createMatrix(int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
        Matrix matrix = new Matrix();

        if (srcWidth != dstWidth || srcHeight != dstHeight) {
            float scaleFactorX = dstWidth / (float) srcWidth;
            float scaleFactorY = dstHeight / (float) srcHeight;
            float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
            matrix.postScale(scaleFactor, scaleFactor);
        }

        matrix.invert(new Matrix());
        return matrix;
    }

    private WritableArray convertRecognitionToWritableArray(List<Classifier.Recognition> recognitions) {
        WritableArray array = new WritableNativeArray();
        for (Classifier.Recognition recognition : recognitions) {
            WritableMap entry = new WritableNativeMap();
            entry.putString("id", recognition.getId());
            entry.putString("name", recognition.getTitle());
            entry.putDouble("confidence", recognition.getConfidence());
            array.pushMap(entry);
        }

        return array;
    }
}
