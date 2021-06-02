package com.capstone.hlife;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import androidx.annotation.Nullable;

import java.util.ArrayList;

public class DataHelper extends SQLiteOpenHelper {
    public DataHelper(@Nullable Context context) {
        super(context, Constants.DB_NAME, null, Constants.DB_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        //Membuat table di db
        db.execSQL(Constants.CREATE_TABLE);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        //Upgrade database jika ada struktur yang berubah pada versi database

        //hapus tabel yang lebih lama jika ada
        db.execSQL("DROP TABLE IF EXISTS "+ Constants.TB_NAME);
        //Buat tabel baru
        onCreate(db);
    }

    //tambah record ke dataase
    public long insertRecord(String kode, String nama, String satuan, String jumlah, String exp,
                             String hrg, byte[] gambar, String addedTime, String updatedTime){
        //get writable database karena kita ingin menuliskan data
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        //id akan terisi otomatis karena kita menggunakan auto_increment

        //masukkan data
        values.put(Constants.K_KODE, kode);
        values.put(Constants.K_NAMA, nama);
        values.put(Constants.K_SATUAN, satuan);
        values.put(Constants.K_JUMLAH, jumlah);
        values.put(Constants.K_EXP, exp);
        values.put(Constants.K_GAMBAR, gambar);
        values.put(Constants.K_HRG, hrg);
        values.put(Constants.K_ADDED_TIMESTAMP, addedTime);
        values.put(Constants.K_UPDATED_TIMESTAMP, updatedTime);

        //memasukkan baris, Ini akan mengembalikan id agar tersimpan dalam record
        long id = db.insert(Constants.TB_NAME, null,values);

        //tutup koneksi
        db.close();

        //mengembalikan nilai id dari record yang sudah dimasukkan
        return id;
    }

    //Update record ke database
    public void updateRecord(String kode, String nama, String satuan, String jumlah, String exp,
                             String hrg, byte[] gambar, String addedTime, String updatedTime){
        //get writable database karena kita ingin menuliskan data
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        //id akan terisi otomatis karena kita menggunakan auto_increment

        //masukkan data
        values.put(Constants.K_KODE, kode);
        values.put(Constants.K_NAMA, nama);
        values.put(Constants.K_SATUAN, satuan);
        values.put(Constants.K_JUMLAH, jumlah);
        values.put(Constants.K_EXP, exp);
        values.put(Constants.K_GAMBAR, gambar);
        values.put(Constants.K_HRG, hrg);
        values.put(Constants.K_ADDED_TIMESTAMP, addedTime);
        values.put(Constants.K_UPDATED_TIMESTAMP, updatedTime);

        //memasukkan baris, Ini akan mengembalikan id agar tersimpan dalam record
        db.update(Constants.TB_NAME, values,Constants.K_KODE + " = ?",new String[]{kode});

        //tutup koneksi
        db.close();
    }

    //get semua data
    public ArrayList<Model> getAllRecords(String orderBy){
        //Query orderBy akan menjalankan perintah untuk mengurutkan data
        //Ini akan mengembalikan list atau record saat kita telah mengembalikan tipe arraylist<Model>

        ArrayList<Model> ArrayList = new ArrayList<>();
        //query untuk memilih record
        String selectQuery = "SELECT * FROM " +Constants.TB_NAME+ " ORDER BY " +orderBy;

        SQLiteDatabase db = this.getWritableDatabase();
        Cursor cursor = db.rawQuery(selectQuery, null);

        //Perulangan melalui semua record dan menambahnya ke List
        if(cursor.moveToFirst()){
            do{
                Model model= new Model(
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_KODE)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_NAMA)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_SATUAN)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_JUMLAH)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_EXP)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_HRG)),
                        cursor.getBlob(cursor.getColumnIndex(Constants.K_GAMBAR)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_ADDED_TIMESTAMP)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_UPDATED_TIMESTAMP))
                );

                //Menambah record ke dalam List
                ArrayList.add(model);
            } while(cursor.moveToNext());
        }

        //Tutup Koneksi
        db.close();

        //mengembakikan nilai list;
        return  ArrayList;
    }

    //search semua data
    public ArrayList<Model> searchRecords(String query){
        //Query orderBy akan menjalankan perintah untuk mengurutkan data
        //Ini akan mengembalikan list atau record saat kita telah mengembalikan tipe arraylist<Model>

        ArrayList<Model> ArrayList = new ArrayList<>();
        //query untuk memilih record
        String selectQuery = "SELECT * FROM " +Constants.TB_NAME+ " WHERE " + Constants.K_NAMA + " LIKE '%" + query + "%'"  ;

        SQLiteDatabase db = this.getWritableDatabase();
        Cursor cursor = db.rawQuery(selectQuery, null);

        //Perulangan melalui semua record dan menambahnya ke List
        if(cursor.moveToFirst()){
            byte[] gambar = cursor.getBlob(6);
            do{
                Model model= new Model(
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_KODE)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_NAMA)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_SATUAN)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_JUMLAH)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_EXP)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_HRG)),
                        cursor.getBlob(cursor.getColumnIndex(Constants.K_GAMBAR)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_ADDED_TIMESTAMP)),
                        "" +cursor.getString(cursor.getColumnIndex(Constants.K_UPDATED_TIMESTAMP))
                );

                //Menambah record ke dalam List
                ArrayList.add(model);

            } while(cursor.moveToNext());
        }

        //Tutup Koneksi
        db.close();

        //mengembakikan nilai list;
        return  ArrayList;
    }

    //Hapus data menggunakan kode
    public void hapusData(String kode){
        SQLiteDatabase db = getWritableDatabase();
        db.delete(Constants.TB_NAME, Constants.K_KODE + " = ?", new String[]{kode});
        db.close();
    }

    //get angka dari record
    public int getRecordsCount(){
        String countQuery = "SELECT * FROM " +Constants.TB_NAME;
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery(countQuery, null);

        int count = cursor.getCount();

        cursor.close();

        return count;
    }
}
