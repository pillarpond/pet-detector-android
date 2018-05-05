package pp.petdetector;

import android.content.Intent;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;

import com.getbase.floatingactionbutton.FloatingActionsMenu;
import com.github.chrisbanes.photoview.PhotoView;

import java.io.FileDescriptor;
import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "petdetector";
    private static final String LABELS_FILE = "pet_label.txt";

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

    private static final int REQUEST_READ_IMAGE = 42;
    private static final int REQUEST_IMAGE_CAPTURE = 43;

    private Classifier detector;

    private PhotoView photoView;
    private ProgressBar progressBar;
    private FloatingActionsMenu menu;

    private String[] modelPaths;
    private String[] modelNames;

    private int checkedItem = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        photoView = findViewById(R.id.image);
        progressBar = findViewById(R.id.progressBar);
        menu = findViewById(R.id.menu);

        Resources resources = getResources();
        TypedArray ta = resources.obtainTypedArray(R.array.models);
        int n = ta.length();

        modelNames = new String[n];
        modelPaths = new String[n];

        for (int i = 0; i < n; ++i) {
            int id = ta.getResourceId(i, 0);
            if (id > 0) {
                modelNames[i] = resources.getStringArray(id)[0];
                modelPaths[i] = resources.getStringArray(id)[1];
            }
        }
        ta.recycle();

        setModel();

        Resources res = getResources();
        int width = (int) res.getDimension(R.dimen.canvas_width);
        int height = (int) res.getDimension(R.dimen.canvas_height);

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        Canvas canvas = new Canvas(bitmap);

        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.LINEAR_TEXT_FLAG | Paint.SUBPIXEL_TEXT_FLAG);
        paint.setColor(Color.WHITE);
        paint.setTextSize(res.getDimension(R.dimen.text_size));

        String str = "Press \"+\" button";
        Rect rect = new Rect();
        paint.getTextBounds(str, 0, str.length(), rect);

        canvas.drawText(str, (width - rect.width()) / 2, height / 2, paint);
        photoView.setImageBitmap(bitmap);
    }

    private void setModel() {
        final Snackbar snackBar = Snackbar.make(
                findViewById(R.id.container),
                "Initializing...",
                Snackbar.LENGTH_INDEFINITE);
        snackBar.show();
        menu.setEnabled(false);

        new Thread(() -> {
            try {
                if (detector != null) detector.close();

                detector = Classifier.create(
                        getAssets(), modelPaths[checkedItem], LABELS_FILE);
                runOnUiThread(snackBar::dismiss);
                menu.setEnabled(true);
            } catch (Exception e) {
                Log.e(TAG, "Exception!!", e);
                finish();
            }
        }).start();
    }

    public void onClick(View view) {
        menu.collapse();
        switch (view.getId()) {
            case R.id.doc_button:
                performFileSearch();
                break;
            case R.id.camera_button:
                dispatchTakePictureIntent();
                break;
            case R.id.change_button:
                changeModel();
                break;
        }
    }

    public void performFileSearch() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");

        startActivityForResult(intent, REQUEST_READ_IMAGE);
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    public void changeModel() {
        new AlertDialog.Builder(this)
                .setTitle("Choose model")
                .setSingleChoiceItems(modelNames, checkedItem, (dialogInterface, i) -> {
                    checkedItem = i;
                    setModel();
                }).show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            final Bitmap bitmap;

            switch (requestCode) {
                case REQUEST_READ_IMAGE:
                    bitmap = getBitmapFromUri(data.getData());
                    break;

                case REQUEST_IMAGE_CAPTURE:
                    Bundle extras = data.getExtras();
                    bitmap = (Bitmap) extras.get("data");
                    break;

                default:
                    bitmap = null;
                    break;
            }

            progressBar.setVisibility(View.VISIBLE);
            photoView.setVisibility(View.INVISIBLE);

            new Thread(() -> {
                try {
                    detect(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Exception!!", e);
                }
            }).start();
        }
    }

    private Bitmap getBitmapFromUri(Uri uri) {
        Bitmap bitmap;

        try {
            ParcelFileDescriptor parcelFileDescriptor =
                    getContentResolver().openFileDescriptor(uri, "r");
            FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
            bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
            parcelFileDescriptor.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception!!", e);
            return null;
        }

        return bitmap;
    }

    private void detect(Bitmap bitmap) {
        if (bitmap == null){
            Snackbar.make(findViewById(R.id.container), "Error occurred", Snackbar.LENGTH_SHORT).show();
            drawResult(null);
            return;
        }

        int srcWidth = bitmap.getWidth();
        int srcHeight = bitmap.getHeight();
        int dstWidth = photoView.getWidth();
        int dstHeight = photoView.getHeight();

        float xScale = (float) dstWidth / srcWidth;
        float yScale = (float) dstHeight / srcHeight;

        float scale = Math.min(xScale, yScale);

        float scaledWidth = scale * srcWidth;
        float scaledHeight = scale * srcHeight;

        Bitmap copyBitmap = Bitmap.createScaledBitmap(bitmap, (int) scaledWidth, (int) scaledHeight, true);
        List<Classifier.Recognition> results = detector.recognizeImage(copyBitmap);

        Canvas canvas = new Canvas(copyBitmap);

        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Style.STROKE);
        float borderWidth = getResources().getDimension(R.dimen.border_width);
        paint.setStrokeWidth(borderWidth);

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.LINEAR_TEXT_FLAG | Paint.SUBPIXEL_TEXT_FLAG);
        textPaint.setColor(Color.YELLOW);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);
        textPaint.setTextSize(getResources().getDimension(R.dimen.text_size));

        Paint backgroundPaint = new Paint();
        backgroundPaint.setColor(Color.RED);
        backgroundPaint.setStyle(Style.FILL);

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                canvas.drawRect(location, paint);

                String str = result.getTitle() + " (" + (int) (result.getConfidence() * 100) + "%)";
                float x = location.left + borderWidth / 2;
                float y = location.bottom - borderWidth / 2;

                Rect bounds = new Rect();
                textPaint.getTextBounds(str, 0, str.length(), bounds);
                bounds.offset((int) x, (int) y);

                canvas.drawRect(bounds, backgroundPaint);
                canvas.drawText(str, x, y, textPaint);
            }
        }

        drawResult(copyBitmap);
    }

    void drawResult(Bitmap bitmap) {
        runOnUiThread(()-> {
            progressBar.setVisibility(View.GONE);
            photoView.setImageBitmap(bitmap);
            photoView.setVisibility(View.VISIBLE);
        });
    }
}
