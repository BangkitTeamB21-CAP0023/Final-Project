package com.capstone.hlife;

public class Constants {
    //Nama Database
    public static final String DB_NAME = "SPAREPART";
    //Versi Database
    public static final int DB_VERSION = 1;
    //Nama Tabel
    public static final String TB_NAME = "SPAREPART";

    //kolom
    public static final String K_KODE = "kode";
    public static final String K_NAMA = "nama";
    public static final String K_SATUAN = "satuan";
    public static final String K_JUMLAH  = "jumlah";
    public static final String K_EXP = "exp";
    public static final String K_GAMBAR = "gambar";
    public static final String K_HRG = "hrg";
    public static final String K_ADDED_TIMESTAMP = "added_time_stamp";
    public static final String K_UPDATED_TIMESTAMP = "updated_time_stamp";

    //Buat Table SQL
    public static final String CREATE_TABLE = "CREATE TABLE "+ TB_NAME +"("
            + K_KODE + " TEXT PRIMARY KEY,"
            + K_NAMA + " TEXT,"
            + K_SATUAN + " TEXT,"
            + K_JUMLAH + " INT,"
            + K_EXP + " INT,"
            + K_GAMBAR +" TEXT,"
            + K_HRG +" TEXT,"
            + K_ADDED_TIMESTAMP +" TEXT,"
            + K_UPDATED_TIMESTAMP +" TEXT"
            + ");";
}
