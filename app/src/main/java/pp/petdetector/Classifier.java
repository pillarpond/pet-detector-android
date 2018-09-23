/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package pp.petdetector;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Generic interface for interacting with different recognition engines.
 */
public class Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    // Only return this many results.
    private static final int MAX_RESULTS = 100 * 10;

    // Config values.
    private String inputName;
    private int width;
    private int height;

    // Pre-allocated buffers.
    private ArrayList<String> labels;
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     */
    static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputWidth,
            final int inputHeight) throws IOException {
        final Classifier c = new Classifier();

        InputStream is = assetManager.open(labelFilename);
        c.labels = new ArrayList<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        final Graph g = c.inferenceInterface.graph();

        c.width = inputWidth;
        c.height = inputHeight;

        c.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation
                inputOp = g.operation(c.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + c.inputName + "'");
        }

        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }
        final Operation outputOp4 = g.operation("num_detections");
        if (outputOp4 == null) {
            throw new RuntimeException("Failed to find output Node 'num_detections'");
        }

        // Pre-allocate buffers.
        c.intValues = new int[inputWidth * inputHeight];
        c.byteValues = new byte[inputWidth * inputHeight * 3];

        c.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        c.outputScores = new float[MAX_RESULTS];
        c.outputLocations = new float[MAX_RESULTS * 4];
        c.outputClasses = new float[MAX_RESULTS];
        c.outputNumDetections = new float[1];
        return c;
    }

    private Classifier() {}

    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, height, width, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);
        Trace.endSection();

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        1,
                        (lhs, rhs) -> {
                            // Intentionally reversed to put high confidence at the head of the queue.
                            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            if (outputClasses[i] < 1) continue;

            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * width,
                            outputLocations[4 * i] * height,
                            outputLocations[4 * i + 3] * width,
                            outputLocations[4 * i + 2] * height);
            pq.add(
                    new Recognition("" + i, labels.get((int) outputClasses[i] - 1), outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    public void close() {
        inferenceInterface.close();
    }
}
