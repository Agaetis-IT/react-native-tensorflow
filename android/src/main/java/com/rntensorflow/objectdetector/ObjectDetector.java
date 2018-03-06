package com.rntensorflow.objectdetector;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import com.facebook.react.bridge.*;
import com.rntensorflow.RNTensorflowInference;
import com.rntensorflow.ResourceManager;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.*;

public class ObjectDetector {

    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    private RNTensorflowInference inference;
    private ResourceManager resourceManager;

    private String[] labels;

    public ObjectDetector(RNTensorflowInference inference, ResourceManager resourceManager, String[] labels) {
        this.inference = inference;
        this.resourceManager = resourceManager;
        this.labels = labels;
    }

    public static ObjectDetector init(
            ReactContext reactContext,
            String modelFilename,
            String labelFilename) throws IOException {

        RNTensorflowInference inference = RNTensorflowInference.init(reactContext, modelFilename);
        ResourceManager resourceManager = new ResourceManager(reactContext);
        String[] labels = resourceManager.loadResourceAsString(labelFilename).split("\\r?\\n");

        return new ObjectDetector(inference, resourceManager, labels);
    }

    public WritableArray detect(final String image,
                                        final String inputName,
                                        final Integer inputSize,
                                        final Integer maxResults,
                                        final Double threshold) {

        String inputNameResolved = inputName != null ? inputName : "input";
        String[] outputNames = new String[]{"detection_boxes", "detection_scores", "detection_classes", "num_detections"};
        Integer maxResultsResolved = maxResults != null ? maxResults : MAX_RESULTS;
        Float thresholdResolved = threshold != null ? threshold.floatValue() : THRESHOLD;

        Bitmap bitmapRaw = loadImage(resourceManager.loadResource(image));

        int inputSizeResolved = inputSize != null ? inputSize : 224;
        int[] intValues = new int[inputSizeResolved * inputSizeResolved];
        float[] floatValues = new float[inputSizeResolved * inputSizeResolved * 3];

        Bitmap bitmap = Bitmap.createBitmap(inputSizeResolved, inputSizeResolved, Bitmap.Config.ARGB_8888);
        Matrix matrix = createMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(), inputSizeResolved, inputSizeResolved);
        final Canvas canvas = new Canvas(bitmap);
        canvas.drawBitmap(bitmapRaw, matrix, null);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
            floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
            floatValues[i * 3 + 2] = val & 0xFF;
        }
        Tensor tensor = Tensor.create(new long[]{1, inputSizeResolved, inputSizeResolved, 3}, FloatBuffer.wrap(floatValues));
        inference.feed(inputNameResolved, tensor);
        inference.run(outputNames, false);

        Map<String, ReadableArray> outputs = new HashMap<>();
        for(String outputName : outputNames) {
            outputs.put(outputName, inference.fetch(outputName));
        }

        List<WritableMap> results = new ArrayList<>();
        for (int i = 0; i < outputs.size(); ++i) {
            if (outputs.get("detection_scores").getDouble(i) > thresholdResolved) {
                WritableMap entry = new WritableNativeMap();
                WritableArray boxes = new WritableNativeArray();

                for(int j = 0; i < outputs.get("detection_boxes").getArray(i).size(); j++) {
                    int coord = outputs.get("detection_boxes").getArray(i).getInt(j);
                    boxes.pushInt(coord);
                }

                entry.putString("id", String.valueOf(i));
                entry.putString("name", labels.length > i ? labels[i] : "unknown");
                entry.putDouble("confidence", outputs.get("detection_scores").getDouble(i));
                entry.putArray("boxes", boxes);
                results.add(entry);
            }
        }

        Collections.sort(results, new Comparator<ReadableMap>() {
            @Override
            public int compare(ReadableMap first, ReadableMap second) {
                return Double.compare(second.getDouble("confidence"), first.getDouble("confidence"));
            }
        });
        int finalSize = Math.min(results.size(), maxResultsResolved);
        WritableArray array = new WritableNativeArray();
        for (int i = 0; i < finalSize; i++) {
            array.pushMap(results.get(i));
        }

        inference.getTfContext().reset();
        return array;
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

}
