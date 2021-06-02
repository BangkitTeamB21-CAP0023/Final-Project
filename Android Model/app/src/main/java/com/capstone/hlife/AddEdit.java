package com.capstone.hlife;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.blogspot.atifsoftwares.circularimageview.CircularImageView;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import java.io.ByteArrayOutputStream;

public class AddEdit extends AppCompatActivity {
    //views
    private CircularImageView profilIv;
    private EditText edtKode,edtNama,edtSatuan,edtJumlah,edtHrg,edtExp;
    private Button btnSimpan, btnCencel;

    //perizinan constants(konstanta)
    private static final int CAMERA_REQUEST_CODE = 100;
    private static final int STORAGE_REQUEST_CODE = 101;

    //konstanta mengambul gambar
    private static final int IMAGE_PICK_CAMERA_CODE=102;
    private static final int IMAGE_PICK_GALLERY_CODE=103;

    //variabel yang akan berisi data untuk disimpan
    private Uri imageUri;
    private String kode1,nama,satuan,jumlah,exp,hrg, addedTime, updatedTime;
    public boolean EditMode = false;

    //DataHelper
    private  DataHelper dbHelper;

    //actionbar
    private ActionBar actionBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_edit);

        //init
        actionBar = getSupportActionBar();

        //Judul
        actionBar.setTitle("Tambah Record");

        //Tombol Kembali
        actionBar.setDisplayShowHomeEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);

        //init views
        profilIv = findViewById(R.id.profilIv);
        edtKode = findViewById(R.id.edtKode);
        edtNama = findViewById(R.id.edtNama);
        edtSatuan = findViewById(R.id.edtSatuan);
        edtJumlah = findViewById(R.id.edtJumlah);
        edtExp = findViewById(R.id.edtExp);
        edtHrg = findViewById(R.id.edtKlr);
        btnSimpan = findViewById(R.id.btnSimpan);
        btnCencel= findViewById(R.id.btnCencel);

        //get data dari intent
        Intent intent = getIntent();
        EditMode = intent.getBooleanExtra("EditMode", false);
        if(EditMode){

            //update data
            actionBar.setTitle("Update Data");

            //get data dari intent
            kode1 = intent.getStringExtra("KODE");
            nama = intent.getStringExtra("NAMA");
            satuan = intent.getStringExtra("SATUAN");
            jumlah = intent.getStringExtra("JUMLAH");
            exp = intent.getStringExtra("EXP");
            hrg = intent.getStringExtra("HRG");
            imageUri = Uri.parse(intent.getStringExtra("GAMBAR"));
            addedTime = intent.getStringExtra("ADDED_TIME");
            updatedTime = intent.getStringExtra("UPDATED_TIME");

            //set data untuk views
            profilIv.setImageResource(R.drawable.pic);
            edtKode.setEnabled(false);
            edtKode.setText(kode1);
            edtNama.setText(nama);
            edtSatuan.setText(satuan);
            edtJumlah.setText(jumlah);
            edtJumlah.setEnabled(false);
            edtExp.setText(exp);
            edtHrg.setText(hrg);
        }
        else{
            //tambah data
            actionBar.setTitle("Tambah Barang");
        }

        //init db helper
        dbHelper = new DataHelper(this);

        //klik image view untuk menampilkan dialog gambar
        profilIv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //menampilkan dialog ambil gambar
                ActivityCompat.requestPermissions(
                        AddEdit.this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        STORAGE_REQUEST_CODE
                );
            }
        });
        //klik btntambah untuk menyimpan data setelah itu data di set kosong/default
        btnSimpan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputData();

                edtKode.setText("");
                edtNama.setText("");
                edtSatuan.setText("");
                edtJumlah.setText("");
                edtExp.setText("");
                edtHrg.setText("");
                profilIv.setImageResource(R.drawable.pic);
                edtKode.requestFocus();
            }
        });
        btnCencel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
    }
    //menkonversi gambar(byte array) ke bitmap
    public static byte[] imageViewToByte(ImageView image) {
        Bitmap bitmap = ((BitmapDrawable)image.getDrawable()).getBitmap();
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }
    private void inputData() {
        //set data yang ada di edit teks menjadi string dan bitmap(gambar)
        kode1 = ""+edtKode.getText().toString().trim();
        nama = ""+edtNama.getText().toString().trim();
        satuan = ""+edtSatuan.getText().toString().trim();
        jumlah = ""+edtJumlah.getText().toString().trim();
        exp = "" +edtExp.getText().toString().trim();
        hrg = ""+edtHrg.getText().toString().trim();
        byte[] gambar = imageViewToByte(profilIv);

        if(EditMode){
            String timestamp = "" +System.currentTimeMillis();
            dbHelper.updateRecord(
                    "" +kode1,
                    "" +nama,
                    "" +satuan,
                    "" +jumlah,
                    "" +exp,
                    "" +hrg,
                    gambar,
                    "" +addedTime, //addedtime tetap sama
                    "" +updatedTime //updatedtime akan diubah
            );
            Toast.makeText(this, "Data Sudah Diupdate", Toast.LENGTH_SHORT).show();
            finish();
        }
        else{
            //buat data baru dan simpan ke database
            String timestamp = ""+System.currentTimeMillis();
            long kode = dbHelper.insertRecord(
                    ""+kode1,
                    ""+nama,
                    ""+satuan,
                    ""+jumlah,
                    "" +exp,
                    "" +hrg,
                    gambar,
                    ""+timestamp,
                    ""+timestamp
            );
            Toast.makeText(this, "Record Berhasil Ditambahkan", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    @Override
    public boolean onSupportNavigateUp() {
        onBackPressed(); //kembali dengan klik tombol kembali
        return super.onSupportNavigateUp();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == STORAGE_REQUEST_CODE){
            if (grantResults.length>0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                //pindah ke gallery
                Intent galleryIntent = new Intent(Intent.ACTION_GET_CONTENT);
                galleryIntent.setType("image/*");
                startActivityForResult(galleryIntent, STORAGE_REQUEST_CODE);
            }
            else {
                Toast.makeText(this, "Anda tidak memiliki perizinan untuk mengakses gallery", Toast.LENGTH_SHORT).show();
            }
            return;
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (requestCode == STORAGE_REQUEST_CODE && resultCode == RESULT_OK){
            Uri imageUri = data.getData();
            CropImage.activity(imageUri)
                    .setGuidelines(CropImageView.Guidelines.ON) //set guidelines pada gambar
                    .setAspectRatio(1,1)//gambar akan berbentuk kotak
                    .start(this);
        }
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE){
            CropImage.ActivityResult result =CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK){
                Uri resultUri = result.getUri();
                //set gambar yang dipilih dari gallery ke image view
                profilIv.setImageURI(resultUri);
            }
            else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE){
                Exception error = result.getError();
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}
