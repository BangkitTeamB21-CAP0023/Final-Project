package com.capstone.hlife;

import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import java.util.Calendar;
import java.util.Locale;

public class RecordData extends AppCompatActivity {
    //views
    private TextView bio, kode, nama, satuan, jumlah,exp, hrg, addedTime, updatedTime;
    private ImageView imageView, profilIv;

    //actionBar
    private ActionBar actionBar;

    //db helper
    private  DataHelper dbHelper;
    private String recordID;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record_data);

        //Setting action bar dengan judul dan tombol kembali
        actionBar = getSupportActionBar();
        actionBar.setTitle("Detail Barang");
        actionBar.setDisplayShowHomeEnabled(true);
        actionBar.setDisplayHomeAsUpEnabled(true);

        //get kode record dari adapter melalui intent
        Intent intent = getIntent();
        recordID = intent.getStringExtra("KODE");

        //init db helper class
        dbHelper = new DataHelper(this);

        //init views
        profilIv = findViewById(R.id.txtprofilIv);
        kode = findViewById(R.id.txtkode);
        nama = findViewById(R.id.txtNama);
        satuan = findViewById(R.id.txtSatuan);
        jumlah = findViewById(R.id.txtJumlah);
        exp = findViewById(R.id.txtExp);
        hrg = findViewById(R.id.txthrg);
        addedTime = findViewById(R.id.txtaddedTime);
        updatedTime = findViewById(R.id.txtupdatedTime);
        imageView = findViewById(R.id.txtprofilIv);

        showRecordDeatils();
    }

    private void showRecordDeatils() {
        //get record deatils

        //query untuk memilih record berdasarkan kode
        String selectQuery = "SELECT * FROM " +Constants.TB_NAME+ " WHERE " +Constants.K_KODE+ " =\"" +recordID+ "\"";
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        Cursor cursor = db.rawQuery(selectQuery, null);

        //Cek semua database pada record tersebut
        if(cursor.moveToFirst()){
            do{
                String kode1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_KODE));
                String nama1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_NAMA));
                String satuan1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_SATUAN));
                String jumlah1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_JUMLAH));
                String exp1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_EXP));
                String hrg1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_HRG));
                byte[] gambar = cursor.getBlob(cursor.getColumnIndex(Constants.K_GAMBAR));
                String addTime = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_ADDED_TIMESTAMP));
                String updatedTime1 = ""+ cursor.getString(cursor.getColumnIndex(Constants.K_UPDATED_TIMESTAMP));

                //Mengubah cat. waktu menjadi hari/bulan/tahun jam:menit dd/mm/yyyy hh:mm contoh 10/12/2019 02:43 PM
                Calendar calendar1 = Calendar.getInstance(Locale.getDefault());
                calendar1.setTimeInMillis(Long.parseLong(addTime));
                String timeAdded = ""+ DateFormat.format("dd/MM/yyyy hh:mm:aa", calendar1);

                Calendar calendar2 = Calendar.getInstance(Locale.getDefault());
                calendar1.setTimeInMillis(Long.parseLong(updatedTime1));
                String timeUpdated = ""+ DateFormat.format("dd/MM/yyyy hh:mm:aa", calendar2);

                //set data
                kode.setText(kode1);
                nama.setText(nama1);
                satuan.setText(satuan1);
                jumlah.setText(jumlah1);
                exp.setText(exp1);
                hrg.setText(hrg1);
                addedTime.setText(timeAdded);
                updatedTime.setText(timeUpdated);
                Bitmap bitmap = BitmapFactory.decodeByteArray(gambar, 0, gambar.length);

                //Jika user tidak melampirkan gambar maka gambar akan null, jadi dalam kasus ini harus di set default
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

    @Override
    public boolean onSupportNavigateUp() {
        onBackPressed();//get activity sebelumnya
        return super.onSupportNavigateUp();
    }
}
