package com.capstone.hlife;

import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

public class RecordBarang extends AppCompatActivity {
    //views
    private TextView bio,kode, nama, satuan, jumlah, exp, hrg, addedTime, updatedTime, Judul, shop;
    private ImageView imageView, profilIv;
    private Button angka,tambah,kurang, btnok, btnCencel;
    int count;
    String kode1;

    //set pilihan(boolean)
    public boolean ShopMode = false;

    //actionBar
    private ActionBar actionBar;

    //db helper
    private  DataHelper dbHelper;
    private String recordID,Tambah, Ubah;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record_barang);

        //Setting action bar dengan judul dan tombol kembali
        actionBar = getSupportActionBar();
        actionBar.setTitle("Ubah Barang");
        actionBar.setDisplayShowHomeEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);

        //get id record dari adapter melalui intent
        final Intent intent = getIntent();
        final Bundle bd = intent.getExtras();
        recordID = intent.getStringExtra("KODE");
        Tambah = intent.getStringExtra("Tambah");
        Ubah = intent.getStringExtra("Ubah");

        //init db helper class
        dbHelper = new DataHelper(this);

        //init views
        profilIv = findViewById(R.id.txtprofilIv);
        kode = findViewById(R.id.txtkode);
        nama = findViewById(R.id.txtNama);
        satuan = findViewById(R.id.txtSatuan);
        jumlah = findViewById(R.id.txtJumlah);
        hrg = findViewById(R.id.txthrg);
        exp = findViewById(R.id.txtExp);
        imageView = findViewById(R.id.txtprofilIv);
        angka = findViewById(R.id.angka);
        tambah = findViewById(R.id.tambah);
        kurang = findViewById(R.id.kurang);
        btnok = findViewById(R.id.btnOK);
        btnCencel = findViewById(R.id.btnCencel);
        Judul = findViewById(R.id.Judul);
        shop = findViewById(R.id.shop);

        //get data dari intent
        Intent intent1 = getIntent();
        ShopMode = intent1.getBooleanExtra("ShopMode", false);
        if(ShopMode){
            Judul.setText("Ubah Makanan");
            shop.setText("Ubah Makanan");
        }
        else{
            Judul.setText("Tambah Jumlah Makanan");
            shop.setText("Tambah Jumlah");
        }
        showRecordDeatils();

        //Button angka, kurang, dan bagi
        tambah.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                count++;
                angka.setText(Integer.toString(count));
            }
        });
        kurang.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(count==0) {
                    angka.setText("0");
                }else{
                    count--;
                    angka.setText(Integer.toString(count));
                }
            }
        });
        //Mengeksekusi perintah untuk menambah/menjual Barang
        btnok.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(intent.hasExtra("Tambah")){
                    tambahjumlah(kode1);
                    Toast.makeText(RecordBarang.this, "Makanan Berhasil di Tambah", Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(RecordBarang.this,MainActivity.class);
                    finish();
                }
                else if(intent.hasExtra("Jual")){
                    kurangjumlah(kode1);
                    Toast.makeText(RecordBarang.this, "Makanan Berhasil di ubah", Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(RecordBarang.this,MainActivity.class);
                    finish();
                }
            }
        });
        btnCencel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
    }

    private void showRecordDeatils() {
        //get record deatils

        //query untuk memilih record berdasarkan id
        String selectQuery = "SELECT * FROM " +Constants.TB_NAME+ " WHERE " +Constants.K_KODE+ " =\"" +recordID+ "\"";

        SQLiteDatabase db = dbHelper.getWritableDatabase();
        Cursor cursor = db.rawQuery(selectQuery, null);

        //Cek semua database pada record tersebut
        if(cursor.moveToFirst()){
            do{
                String kode1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_KODE));
                String nama1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_NAMA));
                String satuan1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_SATUAN));
                String harga1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_HRG));
                String jumlah1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_JUMLAH));
                String exp1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_EXP));
                byte[] gambar = cursor.getBlob(cursor.getColumnIndex(Constants.K_GAMBAR));

                //set data
                kode.setText(kode1);
                nama.setText(nama1);
                hrg.setText(harga1);
                satuan.setText(satuan1);
                jumlah.setText(jumlah1);
                exp.setText(exp1);
                Bitmap bitmap = BitmapFactory.decodeByteArray(gambar, 0, gambar.length);

                //Jika user tidak melampirkan gambar maka imageUri akan null, jadi dalam kasus ini harus di set default
                if(gambar.equals("null")){
                    //tidak ada gambar di record, set default
                    profilIv.setImageResource(R.drawable.pic);
                }
                else{
                    //ada
                    profilIv.setImageBitmap(bitmap);
                }

            }while(cursor.moveToNext());
        }
    }
    private void tambahjumlah(String kode1) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        /*String sql = "UPDATE " +Constants.TB_NAME+ " SET "+Constants.K_JUMLAH+ " = CASE " +
                "WHEN "+Constants.K_JUMLAH+" > "+Integer.parseInt((String) angka.getText())+" THEN "
                +Constants.K_JUMLAH+" - " +Integer.parseInt((String) angka.getText())+" ELSE 0 END " +
                " WHERE "+Constants.K_ID+ " =\"" +recordID+ "\"";
        db.execSQL(sql);
        db.close();*/

        //query untuk menambah jumlah
        String sql = "UPDATE " +Constants.TB_NAME+ " SET " +Constants.K_JUMLAH+ " = "+Constants.K_JUMLAH+" + "
                +Integer.parseInt((String) angka.getText()) +" WHERE "+Constants.K_KODE+ " =\"" +recordID+ "\"";
        db.execSQL(sql);
        db.close();
    }
    private void kurangjumlah(String kode1) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        //query untuk mengurangi jumlah
        String sql = "UPDATE " +Constants.TB_NAME+ " SET "+Constants.K_JUMLAH+ " = CASE " +
                "WHEN "+Constants.K_JUMLAH+" > "+Integer.parseInt((String) angka.getText())+" THEN "
                +Constants.K_JUMLAH+" - " +Integer.parseInt((String) angka.getText())+
                " WHEN "+Constants.K_JUMLAH+" = "+Integer.parseInt((String) angka.getText())+" THEN "
                +Constants.K_JUMLAH+" - " +Integer.parseInt((String) angka.getText())+
                " ELSE "+Constants.K_JUMLAH+" END " +
                " WHERE "+Constants.K_KODE+ " =\"" +recordID+ "\"";
        db.execSQL(sql);
        db.close();
    }
    @Override
    public boolean onSupportNavigateUp() {
        onBackPressed();//get activity sebelumnya
        return super.onSupportNavigateUp();
    }
}

