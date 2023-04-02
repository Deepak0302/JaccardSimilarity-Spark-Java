/**
 * To perform a fuzzy string search across a number of files and print any
 * matching results to standard output. The search term will be given in the command line,
 * together with a path to a directory containing text files to be searched and a similarity threshold.
 */

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CS1003P4 {
    static Logger logger;
    public static void main(String[] args) throws IOException {
        /**
         * Log4j JVM args setting up to filter the logs on console.
         * One can use by passing the below args in command line as well.
         * -Dlog4j.configuration=file:log4j.properties -Dlogfile.name=application -Dspark.yarn.app.container.log.dir=var/logs/
         */
        System.getProperties().setProperty("log4j.configuration","file:log4j.properties");
        System.getProperties().setProperty("logfile.name","applogs");
        System.getProperties().setProperty("spark.yarn.app.container.log.dir","var/logs/");

        logger = Logger.getLogger("CS1003P4");
        String appName = "TestSpark";
        String cluster = "local[*]"; //
        SparkConf conf = new SparkConf()
                .setAppName(appName)
                .setMaster(cluster);

        SparkSession spark = SparkSession
                .builder().config(conf)
                .getOrCreate();

        /**
         * Loading input files in raw zone and getting validated and storing in refined zone
         * args[0] - input files path to be processed
         * args[1] - term criteria/Query string
         * args[2] - Jaccard similarity index threshold to filter the output         *
         */
        String path = args[0];
        String query = args[1];
        int termToken= query.split(" ").length;
        double thresholdValue = Double.valueOf(args[2]);

        JavaRDD<String> rawDataRDD= spark.sparkContext().textFile(path,1).toJavaRDD();
        //Data cleansing as per specification
        JavaRDD<String> filteredDataRDD= rawDataRDD
                .map(x -> x.replaceAll("[^a-zA-z0-9]"," ")
                        .replaceAll("[\\s]+"," ")
                        .toLowerCase());
        JavaRDD<String> wordList = filteredDataRDD.filter(x -> !x.isEmpty())
                .flatMap(x-> Arrays.asList(x.trim().split("[ \\t\\n\\r]"))
                        .iterator());

        List<Row> refinedData = Arrays.asList(RowFactory.create(wordList.collect()));
        StructType schema = new StructType(new StructField[]{
                new StructField(
                        "words", DataTypes.createArrayType(DataTypes.StringType),
                        false, Metadata.empty())
        });
        Dataset<Row> wordDataFrame = spark.createDataFrame(refinedData, schema);
        /**
         * Get the Bigram tokens of words as per the size of term query to compare the index
         */
        NGram ngramTransformer = new NGram().setN(termToken).setInputCol("words").setOutputCol("ngrams");
        Dataset<Row> ngramDataFrame = ngramTransformer.transform(wordDataFrame);
        List<Row> nGramRefindedData= ngramDataFrame.select(ngramDataFrame.col("ngrams")
                .toString().trim()).takeAsList(3);
        for (Row r :nGramRefindedData ) {
            java.util.List<String> ngrams = r.getList(0);
            //String query= "setting sail to the rising wind";
            for (String ngram : ngrams) {
                double derivedValue=0.0d;
                if(ngram.split(" ").length==termToken)
                    derivedValue= calculateJaccardSimilarity(ngram,query);
                if(derivedValue >= thresholdValue ){
                    logger.info( ngram);
                }
            };
        }

    spark.stop();
    }

    /**
     * This method takes two argument left, right
     * left is for to be validated string with term query which right below
     * @param left
     * @param right
     * @return
     */
    private static double calculateJaccardSimilarity(String left, String right) {
        Set<String> leftSetChBigram = getSetOfCharBigram(left);
        Set<String> rightSetChBigram = getSetOfCharBigram(right);
        int leftLength = leftSetChBigram.size();
        int rightLength = rightSetChBigram.size();

        if (leftLength == 0 && rightLength == 0) {
            return 1.0;
        } else if (leftLength != 0 && rightLength != 0) {

            Set<String> intersection = new HashSet<>(leftSetChBigram);
            intersection.retainAll(rightSetChBigram);
            Set<String> union = new HashSet<>(leftSetChBigram);
            union.addAll(rightSetChBigram);
            double jaccardIndex = 1.0d * (double) intersection.size() / union.size();
            return jaccardIndex;
        }else {
            return 0.0;
        }
    }

    /**
     * This is takes String input and create Character Bi-gram token to check the exact string similarity
     * One can alter the code by changing the value from 2 to N character Bi-gram
     * @param gramString
     * @return
     */
    private static Set<String> getSetOfCharBigram(String gramString) {
        Set<String> set = new HashSet<>();
        String baseString= gramString;
        for (int i=0; i<= baseString.length()-2;i++){
            String tempStr= baseString.substring(i,i+2);
            set.add(tempStr);
        }
        return set;
    }
}