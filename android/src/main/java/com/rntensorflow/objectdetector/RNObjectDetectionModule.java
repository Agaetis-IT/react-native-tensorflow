package com.rntensorflow.objectdetector;

import com.facebook.react.bridge.*;
import com.rntensorflow.imagerecognition.ImageRecognizer;

import java.util.HashMap;
import java.util.Map;

public class RNObjectDetectionModule extends ReactContextBaseJavaModule {

    private Map<String, ObjectDetector> objectDetectors = new HashMap<>();
    private ReactApplicationContext reactContext;

    public RNObjectDetectionModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @Override
    public String getName() {
        return "RNObjectDetection";
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
            ObjectDetector objectDetector = ObjectDetector.create(reactContext, model, labels);
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

            ObjectDetector objectDetector = objectDetectors.get(id);
            WritableArray result = objectDetector.detect(image);
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
