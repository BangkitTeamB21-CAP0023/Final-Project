package com.capstone.hlife;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class Adapter extends RecyclerView.Adapter<Adapter.HolderRecord> {
    //variabel
    private Context context;
    private java.util.ArrayList<Model> ArrayList;

    //DB Helper
    DataHelper dbHelper;

    //konstruktor
    public Adapter(Context context, java.util.ArrayList<Model> arrayList) {
        this.context = context;
        ArrayList = arrayList;
        dbHelper = new DataHelper(context);
    }

    @NonNull
    @Override
    public HolderRecord onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        //inflate record
        View view = LayoutInflater.from(context).inflate(R.layout.view_data, parent, false);


        return new HolderRecord(view);
    }

    @Override
    public void onBindViewHolder(@NonNull HolderRecord holder, final int position) {
        //get data, set data, menangani view klik di method ini


        //get data
        Model model = ArrayList.get(position);
        final String kode = model.getKode();
        final String nama = model.getNama();
        final String satuan = model.getSatuan();
        final String jumlah = model.getJumlah();
        final String exp = model.getExp();
        final String hrg = model.getKcal();
        final byte[] gambar = model.getGambar();
        final String addedTime = model.getAddedTime();
        final String updatedTime = model.getUpdatedTime();

        //set data agar terlihat
        holder.kode.setText(kode);
        holder.nama.setText(nama);
        holder.satuan.setText(satuan);
        holder.exp.setText(exp);
        holder.jumlah.setText(jumlah +" "+satuan);
        holder.hrg.setText(hrg);

        Bitmap bitmap = BitmapFactory.decodeByteArray(gambar, 0, gambar.length);
        //Jika user tidak melampirkan gambar maka imageUri akan null, jadi dalam kasus ini harus di set default
        if(gambar.equals("null")){
            //tidak ada gambar di record, set default
            holder.profil.setImageResource(R.drawable.pic);
        }
        else{
            //ada
            holder.profil.setImageBitmap(bitmap);
        }
        //Menangani  item klik untuk melihat detail record
        holder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Mengambil record id untuk ke activity selanjutnya (detail record)
                Intent intent = new Intent(context, RecordData.class);
                intent.putExtra("KODE", kode);
                context.startActivity(intent);
            }
        });

        //Menangani tombol more(menu lain) agar menampilan pilihan edit,hapus dll
        holder.btnMore.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //menampilkan pilihan menu
                showMoreDialog(
                        ""+position,
                        ""+kode,
                        ""+nama,
                        ""+satuan,
                        ""+jumlah,
                        ""+exp,
                        ""+hrg,
                        ""+gambar,
                        ""+addedTime,
                        ""+updatedTime
                );
            }
        });

        //Menangani Tombol Tambah barang
        holder.btntambah_brg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(context, RecordBarang.class);
                intent.putExtra("KODE", kode);
                intent.putExtra("Tambah", "tambah");
                context.startActivity(intent);
                Toast.makeText(context, "Tambah Barang", Toast.LENGTH_SHORT).show();

            }
        });
    }

    private void showMoreDialog(String position, final String kode, final String nama, final String satuan,
                                final String jumlah, final String exp, final String hrg, final String gambar, final String addedTime, final String updatedTime) {
        //Pilihan untuk menampilkan di dialog
        String[] options = {"Edit","Delete"};
        //dialog
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        //Menambahkan item ke dialog
        builder.setItems(options, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //Menangani item klik
                if(which == 0){
                    //Edit jika di klik

                    //Memasukkan data ke values untuk dikirim ke activity tambah
                    Intent intent = new Intent(context, AddEdit.class);
                    intent.putExtra("KODE",kode);
                    intent.putExtra("NAMA",nama);
                    intent.putExtra("SATUAN",satuan);
                    intent.putExtra("JUMLAH",jumlah);
                    intent.putExtra("EXP",exp);
                    intent.putExtra("HRG",hrg);
                    intent.putExtra("GAMBAR",gambar);
                    intent.putExtra("ADDED_TIME",addedTime);
                    intent.putExtra("UPDATED_TIME",updatedTime);
                    intent.putExtra("EditMode", true); //update data yang ada, set true

                    context.startActivity(intent);

                }
                else if(which == 1){
                    //Dihapus jika di klik
                    dbHelper.hapusData(kode);
                    //refresh record dengan menggunakan method onResume
                    ((ListData)context).onResume();
                }
            }
        });
        //Menampilkan dialog
        builder.create().show();

    }
    @Override
    public int getItemCount() {
        return ArrayList.size(); //mengembalikan ukuran dari list/angka dari record
    }

    class HolderRecord extends RecyclerView.ViewHolder{

        //views
        ImageView profil;
        TextView kode,nama,satuan,hrg,jumlah,exp;
        ImageButton btnMore;
        Button btntambah_brg;

        public HolderRecord(@NonNull View itemView) {
            super(itemView);

            //init views
            profil = itemView.findViewById(R.id.r_profil);
            kode = itemView.findViewById(R.id.r_kode);
            nama = itemView.findViewById(R.id.r_nama);
            satuan = itemView.findViewById(R.id.r_satuan);
            hrg = itemView.findViewById(R.id.r_hrg);
            jumlah = itemView.findViewById(R.id.r_jumlah);
            exp = itemView.findViewById(R.id.r_exp);
            btnMore = itemView.findViewById(R.id.r_btnMore);
            btntambah_brg = itemView.findViewById(R.id.btntambah_brg);

        }
    }
}
