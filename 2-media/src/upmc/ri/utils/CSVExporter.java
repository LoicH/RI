package upmc.ri.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;


public class CSVExporter {
	 public static void exportArray(List<Float> a, String filepath){
         try {
             File f = new File(filepath);
             if(!f.exists()){
                 f.createNewFile();
             }
             FileWriter fw = new FileWriter(filepath);
        
            
             for(int i = 0 ; i < a.size() ; i++){
                 fw.write(a.get(i)+"");
                 if( i != a.size() - 1){
                     fw.write(";");
                 }
                
             }
             fw.close();
         } catch (IOException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
         }
        
        
 }

 public static void exportMatrix(double[][] a, String filepath){
     try {
         File f = new File(filepath);
         if(!f.exists()){
             f.createNewFile();
         }
         FileWriter fw = new FileWriter(filepath);
    
         for(int i = 0 ; i < a.length ; i++){
            
             for(int j = 0 ; j < a[i].length; j++){
            
                 fw.write(a[i][j]+"");
                
                 if( j != a[i].length - 1){
                     fw.write(";");
                 }
                
             }
             fw.write("\n");
         }
         fw.close();
     } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
     }
 }

}
