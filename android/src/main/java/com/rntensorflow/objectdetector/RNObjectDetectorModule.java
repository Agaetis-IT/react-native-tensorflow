package com.rntensorflow.objectdetector;

import com.facebook.react.bridge.*;
import com.rntensorflow.imagerecognition.ImageRecognizer;

import java.util.*;

public class RNObjectDetectorModule extends ReactContextBaseJavaModule {

    private Map<String, ObjectDetector> objectDetectors = new HashMap<>();
    private ReactApplicationContext reactContext;

    public RNObjectDetectorModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @Override
    public String getName() {
        return "RNObjectDetector";
    }

    @Override
    public void onCatalystInstanceDestroy() {
        for (String id : objectDetectors.keySet()) {
            this.objectDetectors.remove(id);
        }
    }

    @ReactMethod
    public void initObjectDetector(String id, ReadableMap data, Promise promise) {
        try {
            String model = data.getString("model");
            String labels = data.getString("labels");

            ObjectDetector objectDetector = ObjectDetector.init(reactContext, model, labels);
            objectDetectors.put(id, objectDetector);
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject(e);
        }
    }

    @ReactMethod
    public void detect(String id, ReadableMap data, Promise promise) {
        try {
            String image = data.getString("image");
            String inputName = data.hasKey("inputName") ? data.getString("inputName") : null;
            Integer inputSize = data.hasKey("inputSize") ? data.getInt("inputSize") : null;
            Integer maxResults = data.hasKey("maxResults") ? data.getInt("maxResults") : null;
            Double threshold = data.hasKey("threshold") ? data.getDouble("threshold") : null;

            ObjectDetector imageRecognizer = objectDetectors.get(id);
            WritableArray result = imageRecognizer.detect(image, inputName, inputSize, maxResults, threshold);
            promise.resolve(result);
        } catch (Exception e) {
            promise.reject(e);
        }
    }

    @ReactMethod
    public void close(String id, Promise promise) {
        try {
            this.objectDetectors.remove(id);
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject(e);
        }
    }
}
