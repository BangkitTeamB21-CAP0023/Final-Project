package com.capstone.hlife;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class Adapter2 extends RecyclerView.Adapter<Adapter2.HolderRecord> {
    //variabel
    private Context context;
    private java.util.ArrayList<Model> ArrayList;

    //DB Helper
    DataHelper dbHelper;

    //konstruktor
    public Adapter2(Context context, java.util.ArrayList<Model> arrayList) {
        this.context = context;
        ArrayList = arrayList;
        dbHelper = new DataHelper(context);
    }

    @NonNull
    @Override
    public HolderRecord onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        //inflate record
        View view = LayoutInflater.from(context).inflate(R.layout.view_transaksi, parent, false);


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
        holder.hrg.setText(hrg);
        holder.jumlah.setText(jumlah +" "+satuan);
        holder.exp.setText(exp);

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

        //Menangani Tombol Ubah makanan
        holder.btnjual_brg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(context, RecordBarang.class);
                intent.putExtra("KODE", kode);
                intent.putExtra("Jual", "jual");
                intent.putExtra("ShopMode", true);
                context.startActivity(intent);
                Toast.makeText(context, "Ubah Makanan", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public int getItemCount() {
        return ArrayList.size(); //mengembalikan ukuran dari list/angka dari record
    }

    class HolderRecord extends RecyclerView.ViewHolder{

        //views
        ImageView profil;
        TextView hrg, kode,nama,satuan,jumlah,exp;
        Button btntambah_brg, btnjual_brg;

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
            btntambah_brg = itemView.findViewById(R.id.btntambah_brg);
            btnjual_brg = itemView.findViewById(R.id.btnjual_brg);

        }
    }
}
